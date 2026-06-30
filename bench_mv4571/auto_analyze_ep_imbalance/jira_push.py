#!/usr/bin/env python3
# =============================================================================
# MV-4571/4572 — push EP-imbalance sweep results to Jira (loop + idempotent).
#
# Comment flow per ticket:
#   0) OVERVIEW  — setup, presets used, test matrix, the two analysis directions.
#   1) TOKEN methodology  — [EP_COLLECT] fp8.py instrumentation + how the
#      token-routing imbalance is computed + analyze_tokens.py + sample images.
#   2) TIME  methodology  — MoE comm->compute->comm barrier model + how the
#      MoE-time imbalance/headroom is computed + analyze_time.py + sample images.
#   3) ONE comment PER scenario (config) — token+time numbers, images, conclusion.
#
# Idempotent: a JSON state file records what was already posted, so re-running
# (or --loop) only posts NEW things. Images are uploaded as issue attachments
# (REST) and embedded in the comment via ADF media nodes.
#
# Auth (Jira Cloud): export JIRA_EMAIL + JIRA_API_TOKEN (create at
#   https://id.atlassian.com/manage-profile/security/api-tokens ). Basic auth.
#
# Usage:
#   JIRA_EMAIL=you@moreh.com.vn JIRA_API_TOKEN=xxxx python3 jira_push.py --once
#   ... --logs <dir1>,<dir2>     # scan several run roots (e.g. gpu-5 + gpu-6)
#   ... --loop 120               # re-scan every 120s (push as the sweep finishes)
#   ... --dry-run                # build everything, upload/post nothing
#   [--only MV-4571]
# =============================================================================
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

SITE = "https://moreh.atlassian.net"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_LOGS = os.path.join(SCRIPT_DIR, "logs")
STATE_FILE = os.path.join(SCRIPT_DIR, ".jira_push_state.json")
TOKENS_PY = os.path.join(SCRIPT_DIR, "analyze_tokens.py")
TIME_PY = os.path.join(SCRIPT_DIR, "analyze_time.py")

MODEL_TICKET = {"glm5.2": "MV-4571", "kimi2.6": "MV-4572"}

# --- per-config images, shown as ONE ROW each (ADF mediaGroup). Only files that
#     exist are attached. ---
GROUP_TOKEN_IMGS = ["imbalance_per_layer.png", "load_e2e_per_rank.png",
                    "token_imbalance_hist_all.png", "token_imbalance_hist_prefill_mixed.png",
                    "token_imbalance_hist_decode.png"]
GROUP_TIME_IMGS = ["per_rank_time_breakdown.png", "time_imbalance_hist_all.png",
                   "time_imbalance_hist_prefill.png", "time_imbalance_hist_decode.png"]

# --- methodology sample images (from one representative scenario) ---
TOKEN_METHOD_IMGS = [
    ("token_imbalance_hist_decode.png", "Decode token-routing imbalance (max/min) histogram"),
    ("token_imbalance_hist_prefill_mixed.png", "Prefill/mixed token imbalance (max/min) histogram"),
    ("imbalance_per_layer.png", "Mean token imbalance per MoE layer"),
    ("load_heatmap_layer_x_rank.png", "Cumulative routing load heatmap (layer x rank)"),
]
TIME_METHOD_IMGS = [
    ("@gantt", "Barrier model: gather-end and reduce-scatter-end align across the 8 ranks; "
               "only the MoE compute in between differs per rank"),
    ("time_imbalance_hist_prefill.png", "Prefill MoE-time imbalance (max/min)"),
    ("time_imbalance_hist_decode.png", "Decode MoE-time imbalance (max/min)"),
    ("per_rank_time_breakdown.png", "Per-rank MoE / gather / reduce-scatter time (ms)"),
    ("barrier_dev.png", "Barrier deviation: the ~1% of clusters with no clean sync point"),
]

# The instrumentation added to vllm fp8.py to emit the [EP_COLLECT] lines that
# analyze_tokens.py parses (gated by VLLM_MOREH_EP_LOG=1, requires enforce_eager).
FP8_PATCH_SNIPPET = '''# vllm/model_executor/layers/quantization/fp8.py
# module scope, right after `logger = init_logger(__name__)`:
_EP_COLLECT_IT: dict = {}

def _ep_collect_maybe_log(layer, topk_ids):
    """Log per-global-expert token-routing histogram (topk_ids, pre-all2all)."""
    import os
    if os.environ.get("VLLM_MOREH_EP_LOG", "0") != "1":
        return
    try:
        E = int(getattr(layer, "global_num_experts", 0) or 0)
        if E <= 0 or topk_ids is None:
            return
        name = getattr(layer, "layer_name", None) or "unknown"
        flat = topk_ids.reshape(-1).to(torch.long)
        flat = flat[(flat >= 0) & (flat < E)]
        counts = torch.bincount(flat, minlength=E)[:E].tolist()
        ntok = int(topk_ids.shape[0])
        it = _EP_COLLECT_IT.get(name, 0)
        _EP_COLLECT_IT[name] = it + 1
        logger.info("[EP_COLLECT] layer=%s it=%d ntok=%d E=%d counts=[%s]",
                    name, it, ntok, E, ",".join(map(str, counts)))
    except Exception as e:
        logger.warning("[EP_COLLECT] logging failed: %s", e)

# call site, inside Fp8MoEMethod.apply, just before the moe_kernel dispatch:
        _ep_collect_maybe_log(layer, topk_ids)
        return self.moe_kernel.apply(
            ...
        )
'''

EMAIL = os.environ.get("JIRA_EMAIL", "")
TOKEN = os.environ.get("JIRA_API_TOKEN", "")
DRY = False
# JIRA_IMG_ALL=1 -> attach image rows to EVERY config (use when the sweep is small, e.g. kimi 8 cfgs).
# default (0) -> only the group's main config carries images (keeps total attachments low for big sweeps).
IMG_ALL = os.environ.get("JIRA_IMG_ALL", "0") == "1"


# ----------------------------- REST (via curl) -----------------------------
def _curl(args):
    r = subprocess.run(["curl", "-sS", "--fail-with-body", "-u", f"{EMAIL}:{TOKEN}"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr or r.stdout}")
    return r.stdout


def resolve_media_uuid(att_id, tries=6):
    """Map a numeric attachment id to its Media Services file UUID (needed by ADF
       media nodes). The content endpoint 30x-redirects to .../file/<uuid>/...
       It can briefly 404 right after upload, so retry with backoff."""
    last = ""
    for i in range(tries):
        r = subprocess.run(["curl", "-sS", "-D", "-", "-o", "/dev/null", "-u",
                            f"{EMAIL}:{TOKEN}",
                            f"{SITE}/rest/api/3/attachment/content/{att_id}"],
                           capture_output=True, text=True)
        m = re.search(r"/file/([0-9a-fA-F-]{36})", r.stdout)
        if m:
            return m.group(1)
        last = r.stdout[:200]
        time.sleep(1.5)
    raise RuntimeError(f"cannot resolve media uuid for attachment {att_id}: {last}")


def upload_attachment(key, path):
    """Attach a file to the issue; return its Media Services UUID (for ADF media)."""
    if DRY:
        return f"DRY:{os.path.basename(path)}"
    out = _curl(["-X", "POST", "-H", "X-Atlassian-Token: no-check",
                 "-F", f"file=@{path}", f"{SITE}/rest/api/3/issue/{key}/attachments"])
    return resolve_media_uuid(json.loads(out)[0]["id"])


def post_comment(key, adf, parent_id=None):
    """Post a comment. With parent_id, post it as a threaded reply (Jira's
       undocumented `parentId` field on the comment POST)."""
    if DRY:
        return "DRY"
    payload = {"body": adf}
    if parent_id:
        payload["parentId"] = str(parent_id)
    body = json.dumps(payload)
    out = _curl(["-X", "POST", "-H", "Content-Type: application/json",
                 "--data", body, f"{SITE}/rest/api/3/issue/{key}/comment"])
    return json.loads(out)["id"]


def get_comment_body(key, cid):
    """Fetch a comment's current ADF body (so we can patch text without touching media)."""
    out = _curl(["-H", "Accept: application/json",
                 f"{SITE}/rest/api/3/issue/{key}/comment/{cid}"])
    return json.loads(out)["body"]


def put_comment(key, cid, adf):
    """Replace an existing comment's body in place (PUT). Media UUIDs already in `adf`
       are reused as-is — no re-upload, so no orphan attachments are created."""
    if DRY:
        return
    body = json.dumps({"body": adf})
    _curl(["-X", "PUT", "-H", "Content-Type: application/json",
           "--data", body, f"{SITE}/rest/api/3/issue/{key}/comment/{cid}"])


# ----------------------------- ADF builders -----------------------------
def _text(s, marks=None):
    n = {"type": "text", "text": s}
    if marks:
        n["marks"] = marks
    return n


def heading(s, level=3):
    return {"type": "heading", "attrs": {"level": level}, "content": [_text(s)]}


def para(s):
    return {"type": "paragraph", "content": ([_text(s)] if s else [])}


def caption(s):
    return {"type": "paragraph", "content": [_text(s, [{"type": "em"}])]}


def code(s, lang="python"):
    return {"type": "codeBlock", "attrs": {"language": lang}, "content": ([_text(s)] if s else [])}


def bullets(items):
    return {"type": "bulletList",
            "content": [{"type": "listItem", "content": [para(i)]} for i in items]}


def png_dims(path):
    """(w, h) from the PNG IHDR, or (None, None)."""
    try:
        import struct
        b = open(path, "rb").read(24)
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", b[16:24])
    except Exception:
        pass
    return None, None


def media(att_id, w=None, h=None):
    """A single centered image. Width/height (when known) avoid the large blank gap
       Jira reserves for dimensionless media."""
    m = {"type": "media", "attrs": {"id": att_id, "type": "file", "collection": ""}}
    ms = {"type": "mediaSingle", "attrs": {"layout": "center"}}
    if w and h:
        m["attrs"]["width"] = int(w); m["attrs"]["height"] = int(h)
        ms["attrs"]["width"] = min(int(w), 680); ms["attrs"]["widthType"] = "pixel"
    ms["content"] = [m]
    return ms


def media_group(att_ids):
    """A row of images (ADF mediaGroup) — renders thumbnails side by side."""
    return {"type": "mediaGroup",
            "content": [{"type": "media", "attrs": {"id": a, "type": "file", "collection": ""}}
                        for a in att_ids]}


def figures(items):
    """items: list of (att_id, caption_str, w, h) -> [caption, media, ...]."""
    out = []
    for aid, cap, w, h in items:
        if cap:
            out.append(caption(cap))
        out.append(media(aid, w, h))
    return out


def rule():
    return {"type": "rule"}


def expand(title, blocks):
    """Collapsible section (ADF expand). Keeps long code out of the way."""
    return {"type": "expand", "attrs": {"title": title}, "content": blocks}


def _cell(s, header=False):
    return {"type": "tableHeader" if header else "tableCell", "attrs": {},
            "content": [para(str(s))]}


def table(header, rows):
    """ADF table: header list + list of row lists (all stringified)."""
    content = [{"type": "tableRow", "content": [_cell(h, header=True) for h in header]}]
    for r in rows:
        content.append({"type": "tableRow", "content": [_cell(c) for c in r]})
    return {"type": "table", "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
            "content": content}


def doc(blocks):
    return {"type": "doc", "version": 1, "content": blocks}


def read_script(path, max_chars=28000):
    try:
        s = open(path).read()
    except OSError:
        return "(script not found)"
    return s if len(s) <= max_chars else s[:max_chars] + "\n# ... (truncated; see repo) ...\n"


# ----------------------------- presets -----------------------------
def mtp_of(preset):
    p = preset.lower()
    if "nomtp" in p:
        return "MTP 0"
    m = re.search(r"mtp(\d+)", p)
    return f"MTP {m.group(1)}" if m else preset


def preset_yaml_path(log_root, preset_dir):
    return os.path.join(log_root, "presets", "glm5.2", "dp8ep8", f"{preset_dir}.yaml")


def summarize_preset(preset_dir, yaml_path):
    """One bullet string summarizing a preset's salient config."""
    txt = ""
    try:
        txt = open(yaml_path).read()
    except OSError:
        return f"{preset_dir} ({mtp_of(preset_dir)}): preset file not found"

    def grab(key):
        m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", txt, re.M)
        return m.group(1).strip().strip("'\"") if m else None

    spec = grab("speculative_config")
    mtp = "0"
    if spec:
        m = re.search(r"num_speculative_tokens['\"]?\s*:\s*(\d+)", spec)
        if m:
            mtp = m.group(1)
    eager = grab("enforce_eager") or "false"
    bs = grab("block_size")
    quant = grab("quantization")
    kv = grab("kv_cache_dtype")
    gmu = grab("gpu_memory_utilization")
    return (f"{preset_dir}: MTP {mtp} (speculative={'mtp' if mtp != '0' else 'off'}), "
            f"enforce_eager={eager}, block_size={bs}, quant={quant}, "
            f"kv_cache={kv}, gpu_mem_util={gmu}")


# ----------------------------- methodology docs -----------------------------
def overview_doc(ticket, preset_lines, isls, concs):
    return doc([
        heading("EP-imbalance analysis — setup & approach", 2),
        para("Target: DP8 / EP8 / TP1 on MI300 (ROCm + AITER), GLM-5.2-FP8, dataset "
             "longbenchv2-custom. GLM-5.2: 256 experts, top-k 8, 78 layers, "
             "first_k_dense_replace=3 -> 75 MoE layers; EP8 -> 32 experts/rank."),
        heading("Presets run", 3),
        bullets(preset_lines),
        para("Common config across presets: tensor_parallel_size=1, data_parallel_size=8, "
             "enable_expert_parallel=true, fp8 per-block->per-token quant, kv_cache fp8_e4m3, "
             "cudagraph FULL_DECODE_ONLY. EP backend = AgRs (all_gatherv dispatch + "
             "reduce_scatterv combine), synchronous (supports_async=false) -> a barrier."),
        heading("Test matrix", 3),
        bullets([
            f"ISL: {', '.join(isls)} (OSL 1024 for 8k, 500 otherwise), rate = inf.",
            f"concurrency: {', '.join(concs)}.",
            "Each (preset, ISL, conc) is served twice (one token phase, one time phase) so "
            "each measurement is clean; prefix caching off -> every conc measured cold.",
        ]),
        heading("Two analysis directions", 3),
        bullets([
            "by #tokens: per-(layer, step) routing load before all2all -> max/min load per rank. "
            "A proxy that overstates the real cost (ignores tiling/padding + fixed overhead).",
            "by time: MoE-compute span per rank from profiler traces -> imbalance and headroom "
            "if balanced. This is the ground truth.",
        ]),
        para("Methodology and per-scenario results follow in the next comments."),
    ])


def token_method_doc(att_caps):
    b = [
        heading("EP imbalance by #tokens — routing load before all2all", 2),
        para("Before the all-gather dispatch of each MoE (layer, step), every rank logs the "
             "histogram of its tokens' top-8 expert ids over all 256 experts. This is emitted by "
             "an instrumentation hook in vllm fp8.py (Fp8MoEMethod.apply), gated by "
             "VLLM_MOREH_EP_LOG=1 and requiring enforce_eager (under cudagraph the side effect is "
             "captured once, not replayed per step)."),
        heading("Instrumentation: [EP_COLLECT] in fp8.py", 3),
        expand("fp8.py instrumentation patch (click to expand)",
               [code(FP8_PATCH_SNIPPET)]),
        para("Line format: [EP_COLLECT] layer=<name> it=<k> ntok=<local_tokens> E=256 "
             "counts=[c0,...,c255]."),
        heading("How imbalance is computed", 3),
        bullets([
            "Aggregate: sum the 8 ranks' histograms per (layer, step) -> global per-expert "
            "routing count g[256].",
            "Linear placement: rank r owns experts [32r, 32r+32). load[r] = sum of g over r's 32 "
            "experts = #token-routings rank r must compute.",
            "imbalance per (layer, step) = max_r(load) / min_r(load).",
            "phase split = auto: num_tokens is bimodal (decode = small, prefill = thousands) "
            "with a wide empty gap between; cut at the geometric midpoint of the largest gap on "
            "a log axis. Robust for both MTP and non-MTP (a fixed 4x/6x*concurrency cuts into the "
            "MTP decode cluster, which sits at ~6-15x concurrency).",
            "Caveat: token-ratio is a proxy; it overstates the real impact. The TIME analysis is "
            "the ground truth.",
        ]),
        heading("Script: analyze_tokens.py", 3),
        para("Main blocks: parse_and_aggregate (regex [EP_COLLECT], normalize step = it - min_it "
             "per (rank,layer), sum counts -> global g per (layer,step)); build_steps_df + "
             "rank_loads (split g into 8 contiguous expert-blocks, max/min and max/avg per step; "
             "auto phase by 4*concurrency); plot_imbalance_hist (all/prefill/decode); "
             "plot_load_distribution + plot_per_layer_load (per-expert / per-rank load, "
             "layer x {expert,rank} heatmaps); plot_per_layer_imbalance (mean imbalance per layer "
             "+ decode layer x step heatmap); export (steps_imbalance.parquet, "
             "expert_dist_global.npy, summary_tokens.json)."),
        expand("analyze_tokens.py — full source (click to expand)",
               [code(read_script(TOKENS_PY))]),
    ]
    if att_caps:
        b.append(heading("Sample output", 3))
        b += figures(att_caps)
    return doc(b)


def time_method_doc(att_caps):
    b = [
        heading("EP imbalance by TIME — MoE comm -> compute -> comm (barrier model)", 2),
        para("MoE pattern on DP8/EP8: communication (all-gather) -> MoE compute -> communication "
             "(reduce-scatter). Each (layer, step) forms one such cluster on every rank; the 8 "
             "ranks run lockstep, so the k-th cluster on all 8 ranks is the same (layer, step). "
             "We map clusters by order of appearance (cluster k of rank0 <-> cluster k of rank1 "
             "... rank7)."),
        para("Expected (barrier) behavior: all-gather and reduce-scatter are synchronous "
             "collectives, so per cluster the gather-END of the 8 ranks ~coincide (one barrier -> "
             "every rank's MoE starts together), and the reduce-scatter-END of the 8 ranks "
             "~coincide (one barrier). Only the MoE compute in between differs per rank (different "
             "load) -> a rank with longer MoE starts reduce-scatter later, but all finish "
             "reduce-scatter together (fast ranks wait for the slowest)."),
        para("Per mapped cluster we record, for all 8 ranks: time(start,end,dur) of comm_gather; "
             "time(start,end,dur) of comm_reduce_scatter; moe_time = reduce_scatter.start - "
             "gather.end."),
        para("Phase split = cudagraph (general, used for both MTP and non-MTP): with cudagraph "
             "capture, decode steps are graph-replayed (one hipGraphLaunch replays the whole graph "
             "-> the MoE GPU kernels have NO per-kernel CPU-launch flow arrow, ac2g), while prefill "
             "runs eager (each kernel has its own arrow). A cluster is decode if most of its GPU "
             "kernels lack a CPU-launch arrow, else prefill."),
        para("imbalance per cluster = max_moe_time(8 ranks) / min_moe_time(8 ranks). "
             "Headroom = 1 - sum(mean_r moe) / sum(max_r moe): reduce-scatter is a barrier so the "
             "critical path is the sum of per-step max; balancing redistributes the same total "
             "work to the mean."),
        heading("Findings on the trace set (~77k clusters, noMTP 8k)", 3),
        bullets([
            "~99% of clusters match the barrier model (clean gather-end / reduce-scatter-end sync "
            "across the 8 ranks).",
            "Time imbalance sits around 1.6 (max/min).",
            "~1% of clusters do not show a clean sync point (gather-end / reduce-scatter-end not "
            "aligned across the 8 ranks) — see the barrier-deviation plot.",
        ]),
        heading("Script: analyze_time.py", 3),
        para("Main blocks: verify_schema (CHECK1-4: trace header/kernel names, shared clock across "
             "ranks, #comm equal across the 8 ranks = mapping valid); parse_rank_pairs (comm-bounded "
             "parse + collect the ac2g flow per cluster -> cg_phase = cudagraph label); "
             "verify_phase_method (1d: cudagraph vs kernel match%); phase_masks (choose split, "
             "default cudagraph); per_rank_table "
             "(MoE/gather/rscat ms per rank + breakdown.png); compute_metrics "
             "(critical=Sum max, balanced=Sum mean, headroom, max/min); plot_time_imbalance_hist "
             "(all/prefill/decode); plot_gantt (gather|moe|reduce-scatter with the two barrier "
             "lines); verify_barrier_rel/abs (barrier_dev*.png); export (trace_pairs.npz, "
             "summary_time.json)."),
        expand("analyze_time.py — full source (click to expand)",
               [code(read_script(TIME_PY))]),
    ]
    if att_caps:
        b.append(heading("Sample output", 3))
        b += figures(att_caps)
    return doc(b)


# ----------------------------- per-scenario doc -----------------------------
def parse_scenario(name):  # e.g. 8k_rinf_c8
    isl = name.split("_")[0]
    conc = name.split("_c")[-1] if "_c" in name else "?"
    rate = "inf"
    if "_r" in name:
        rate = name.split("_r")[1].split("_")[0].replace("p", ".")
    return isl, rate, conc


def fnum(x, n=2):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def conclusion_text(tok, tim):
    """The one-paragraph 'Conclusion:' string for a config. The MoE-compute vs comm
       relationship is config-dependent, so the operator (<< / >> / ~) and the narrative
       are derived from the actual numbers — never hard-coded. End-to-end MoE-layer time
       ~ comm + critical-path(compute); balancing saves (critical - balanced) ms, i.e.
       headroom_pct of the compute path but only (critical-balanced)/(comm+critical) of the
       whole MoE layer."""
    parts = []
    if tok:
        dec = tok.get("decode", {})
        parts.append(
            f"per-step decode token imbalance ~{fnum(dec.get('mean', 0))} while cumulative load is "
            f"~{fnum(tok.get('e2e_rank_max_over_min', 0))} (balanced on aggregate, skewed per step)")
    if tim:
        dec = tim.get("decode", {})
        hp = float(tim.get("headroom_pct", 0) or 0)
        crit = float(tim.get("critical_path_ms", 0) or 0)
        bal = float(tim.get("balanced_ms", 0) or 0)
        comm = float(tim.get("comm_ms", 0) or 0)
        layer = comm + crit
        e2e = ((crit - bal) / layer * 100) if layer > 0 else 0.0
        if comm > 1.5 * crit:
            rel, tail = "<<", (f"comm dominates, so the ~{fnum(hp, 1)}% compute headroom shrinks to "
                               f"~{fnum(e2e, 1)}% of the MoE layer (comm+compute) end-to-end")
        elif crit > 1.5 * comm:
            rel, tail = ">>", (f"compute dominates, so most of that headroom carries through "
                               f"(~{fnum(e2e, 1)}% of the MoE layer end-to-end)")
        else:
            rel, tail = "~", (f"compute and comm are comparable, so the headroom maps to "
                              f"~{fnum(e2e, 1)}% of the MoE layer end-to-end")
        parts.append(
            f"MoE-time imbalance ~{fnum(dec.get('maxmin_mean', 0))}; balancing yields "
            f"~{fnum(hp, 1)}% MoE-compute headroom; MoE-compute ({crit:.0f} ms) {rel} "
            f"comm ({comm:.0f} ms) — {tail}")
    return ("Conclusion: " + "; ".join(parts) + ".") if parts else ""


def scenario_blocks(label, model, preset, name, tok, tim, tok_ids, tim_ids):
    """One config as its own set of blocks. Token images and time images each shown
       as ONE ROW (mediaGroup)."""
    isl, rate, conc = parse_scenario(name)
    b = [heading(f"[{label}] {mtp_of(preset)} | ISL {isl} | conc {conc} | rate {rate}", 3)]

    if tok:
        dec, pre = tok.get("decode", {}), tok.get("prefill_mixed", {})
        b.append(para("By #tokens (phase split = auto):"))
        b.append(bullets([
            f"ranks={tok.get('ranks')} experts={tok.get('experts')} topk={tok.get('topk')} "
            f"experts/rank={tok.get('experts_per_rank')} steps={tok.get('n_steps')}",
            f"decode imbalance max/min: mean {fnum(dec.get('mean',0))}, median "
            f"{fnum(dec.get('median',0))}, p99 {fnum(dec.get('p99',0))} (n={dec.get('n')})",
            f"prefill/mixed imbalance max/min: mean {fnum(pre.get('mean',0))}, p99 "
            f"{fnum(pre.get('p99',0))} (n={pre.get('n')})",
            f"cumulative per-rank load max/min = {fnum(tok.get('e2e_rank_max_over_min',0))}",
        ]))
        if tok_ids:
            b.append(media_group(tok_ids))

    if tim:
        dec = tim.get("decode", {})
        hp = float(tim.get("headroom_pct", 0) or 0)
        crit = float(tim.get("critical_path_ms", 0) or 0)
        bal = float(tim.get("balanced_ms", 0) or 0)
        comm = float(tim.get("comm_ms", 0) or 0)
        ph = tim.get("phase", {})
        method = tim.get("phase_method", "cudagraph")
        b.append(para(f"By time (phase split = {method}):"))
        b.append(bullets([
            f"clusters={tim.get('clusters')} (prefill {ph.get('prefill','?')} / "
            f"decode {ph.get('decode','?')})",
            f"decode MoE imbalance max/min: mean {fnum(dec.get('maxmin_mean',0))}, "
            f"p99 {fnum(dec.get('maxmin_p99',0))}",
            f"critical-path (Sum max) = {crit:.0f} ms | balanced (Sum mean) = {bal:.0f} ms | "
            f"comm = {comm:.0f} ms",
            f"MoE-compute headroom if balanced = {fnum(hp,1)} %",
        ]))
        if tim_ids:
            b.append(media_group(tim_ids))

    ct = conclusion_text(tok, tim)
    if ct:
        b.append(para(ct))
    return b


ISL_ORDER = {"8k": 0, "10k": 1, "100k": 2, "1M": 3}


def config_doc(label, is_main, n_total, idx, model, preset, name, tok, tim, tok_ids, tim_ids):
    """One comment for a single config. The first config of a group (is_main) gets
       a short group header; the rest are sequential follow-up comments."""
    b = []
    if is_main:
        b.append(heading(f"Per-config results — {label}", 2))
        b.append(para(f"One comment per configuration ({n_total} total in this group). "
                      "Each shows token-routing imbalance (row 1) and MoE-time imbalance / "
                      "headroom (row 2). Full methodology is in the earlier comments."))
    else:
        b.append(para(f"({label} — config {idx + 1}/{n_total})"))
    b += scenario_blocks(label, model, preset, name, tok, tim, tok_ids, tim_ids)
    return doc(b)


def summary_table_docs(ticket, scenarios):
    """Two final comments: #tokens and #time. Each splits into 2 tables (non-MTP, MTP),
       with the full all/prefill/decode imbalance columns."""
    members = [s for s in scenarios.values()
               if s["ticket"] == ticket and (s["token"] or s["time"])]

    def ckey(s):
        isl, _, conc = parse_scenario(s["name"])
        return (ISL_ORDER.get(isl, 9), int(conc) if conc.isdigit() else 0)

    def split(gk):
        want = "MTP 0" if gk == "nomtp" else None
        return sorted([s for s in members
                       if (mtp_of(s["preset"]) == "MTP 0") == (gk == "nomtp")], key=ckey)

    def tok_rows(group):
        rows = []
        for s in group:
            if not s["token"]:
                continue
            isl, _, conc = parse_scenario(s["name"])
            t = json.load(open(s["token"]["summary"]))
            a, p, d = t.get("all", {}), t.get("prefill_mixed", {}), t.get("decode", {})
            rows.append([f"{isl} c{conc}", t.get("n_steps"),
                         fnum(a.get("mean")), fnum(p.get("mean")), fnum(d.get("mean")),
                         fnum(d.get("p99")), fnum(t.get("e2e_rank_max_over_min"))])
        return rows

    def tim_rows(group):
        rows = []
        for s in group:
            if not s["time"]:
                continue
            isl, _, conc = parse_scenario(s["name"])
            t = json.load(open(s["time"]["summary"]))
            a, p, d = t.get("all", {}), t.get("prefill", {}), t.get("decode", {})
            ph = t.get("phase", {})
            rows.append([f"{isl} c{conc}", t.get("clusters"), ph.get("prefill"), ph.get("decode"),
                         fnum(a.get("maxmin_mean")), fnum(p.get("maxmin_mean")), fnum(d.get("maxmin_mean")),
                         fnum(d.get("maxmin_p99")), fnum(t.get("headroom_pct"), 1),
                         fnum(t.get("critical_path_ms"), 0), fnum(t.get("comm_ms"), 0)])
        return rows

    TOK_HDR = ["config", "steps", "imb all (mean)", "imb prefill (mean)", "imb decode (mean)",
               "imb decode (p99)", "cum load (max/min)"]
    TIM_HDR = ["config", "clusters", "prefill (n)", "decode (n)", "imb all (mean)",
               "imb prefill (mean)", "imb decode (mean)", "imb decode (p99)", "headroom (%)",
               "critical (ms)", "comm (ms)"]

    tok_doc = doc([
        heading("Summary — EP imbalance by #tokens (all configs)", 2),
        para("imb = per-step routing-load imbalance = max/min over the 8 ranks; (mean)/(p99) = "
             "aggregated over the steps of that phase. 'all' = over ALL steps (independent of the "
             "prefill/decode split). phase split = auto. Token-ratio is a proxy (overstates real "
             "cost) — see the TIME summary for ground truth."),
        heading("non-MTP (MTP 0)", 3), table(TOK_HDR, tok_rows(split("nomtp"))),
        heading("MTP (speculative)", 3), table(TOK_HDR, tok_rows(split("mtp"))),
    ])
    tim_doc = doc([
        heading("Summary — EP imbalance by TIME (all configs)", 2),
        para("imb = per-cluster MoE-compute span imbalance = max/min over the 8 ranks; "
             "(mean)/(p99) = aggregated over the clusters of that phase. 'all', headroom, critical "
             "and comm are over ALL clusters (independent of the prefill/decode split). "
             "phase split = cudagraph. headroom = 1 - Sum(mean)/Sum(max) (reduce-scatter barrier -> "
             "critical path = Sum of per-step max). End-to-end MoE-layer time ~ critical + comm, so the "
             "realized gain ~ headroom x critical/(critical+comm): compare the critical vs comm columns "
             "per row (compute-bound rows keep most of the headroom; comm-bound rows keep less)."),
        heading("non-MTP (MTP 0)", 3), table(TIM_HDR, tim_rows(split("nomtp"))),
        heading("MTP (speculative)", 3), table(TIM_HDR, tim_rows(split("mtp"))),
    ])
    return tok_doc, tim_doc


# ----------------------------- scan + state -----------------------------
def scan(log_dirs):
    """Group token+time analyses per (ticket, model, preset, name). Return
       (scenarios dict, presets-per-ticket dict)."""
    scenarios = {}
    presets = {}
    for logs_dir in log_dirs:
        if not os.path.isdir(logs_dir):
            continue
        log_root = os.path.normpath(os.path.join(logs_dir, ".."))  # logs/ -> repo root
        for run in sorted(os.listdir(logs_dir)):
            runp = os.path.join(logs_dir, run)
            if not os.path.isdir(runp):
                continue
            for model in os.listdir(runp):
                if model not in MODEL_TICKET:
                    continue
                ticket = MODEL_TICKET[model]
                mp = os.path.join(runp, model)
                for preset in os.listdir(mp):
                    pp = os.path.join(mp, preset)
                    if not os.path.isdir(pp):
                        continue
                    presets.setdefault(ticket, {}).setdefault(preset, log_root)
                    for name in os.listdir(pp):
                        base = os.path.join(pp, name)
                        key = (ticket, model, preset, name)
                        s = scenarios.setdefault(key, {"ticket": ticket, "model": model,
                                                        "preset": preset, "name": name,
                                                        "log_root": log_root,
                                                        "token": None, "time": None})
                        for phase, rel in (("token", "tokens/analysis/summary_tokens.json"),
                                           ("time", "time/analysis/summary_time.json")):
                            sp = os.path.join(base, rel)
                            if os.path.isfile(sp):
                                s[phase] = {"summary": sp, "analysis": os.path.dirname(sp)}
    return scenarios, presets


def find_repr(scenarios, ticket, phase):
    """Pick a representative analysis dir for the methodology images.
       Prefer noMTP / 8k, then any with that phase."""
    cands = [s for s in scenarios.values() if s["ticket"] == ticket and s[phase]]

    def rank(s):
        return (0 if "nomtp" in s["preset"].lower() else 1,
                0 if s["name"].startswith("8k") else 1,
                s["name"])
    cands.sort(key=rank)
    return cands[0][phase]["analysis"] if cands else None


def collect_imgs(analysis_dir, spec):
    """spec: list of (filename|'@gantt', caption). Upload existing files; return
       list of (path, caption) so the caller can upload + embed."""
    out = []
    for fn, cap in spec:
        if fn == "@gantt":
            g = sorted(glob.glob(os.path.join(analysis_dir, "gantt", "gantt_*.png")))
            if g:
                out.append((g[0], cap))
            continue
        p = os.path.join(analysis_dir, fn)
        if os.path.isfile(p):
            out.append((p, cap))
    return out


def upload_caps(ticket, path_caps):
    """For methodology figures: (uuid, caption, w, h) with PNG dims (no gap)."""
    out = []
    for p, cap in path_caps:
        w, h = png_dims(p)
        out.append((upload_attachment(ticket, p), cap, w, h))
    return out


def collect_paths(analysis_dir, names):
    """Existing file paths for a row of images (mediaGroup)."""
    return [os.path.join(analysis_dir, n) for n in names
            if os.path.isfile(os.path.join(analysis_dir, n))]


def upload_group(ticket, paths):
    """Upload a row of images, return their media UUIDs (for media_group)."""
    return [upload_attachment(ticket, p) for p in paths]


def load_state():
    if os.path.isfile(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"method": {}, "scenarios": {}}


def save_state(st):
    json.dump(st, open(STATE_FILE, "w"), indent=2)


# ----------------------------- main pass -----------------------------
def ensure_method(ticket, scenarios, presets, st):
    m = st.setdefault("method", {}).setdefault(ticket, {})

    if not m.get("overview"):
        pmap = presets.get(ticket, {})
        plines = [summarize_preset(p, preset_yaml_path(root, p))
                  for p, root in sorted(pmap.items())]
        isls = sorted({s["name"].split("_")[0] for s in scenarios.values()
                       if s["ticket"] == ticket},
                      key=lambda x: {"8k": 0, "10k": 1, "100k": 2, "1M": 3}.get(x, 9))
        concs = sorted({parse_scenario(s["name"])[2] for s in scenarios.values()
                        if s["ticket"] == ticket}, key=lambda x: int(x) if x.isdigit() else 0)
        print(f"[{ticket}] overview comment ...")
        m["overview"] = post_comment(ticket, overview_doc(ticket, plines, isls, concs))
        save_state(st)

    if not m.get("token"):
        rd = find_repr(scenarios, ticket, "token")
        caps = upload_caps(ticket, collect_imgs(rd, TOKEN_METHOD_IMGS)) if rd else []
        print(f"[{ticket}] token methodology ({len(caps)} imgs) ...")
        m["token"] = post_comment(ticket, token_method_doc(caps))
        save_state(st)

    if not m.get("time"):
        rd = find_repr(scenarios, ticket, "time")
        caps = upload_caps(ticket, collect_imgs(rd, TIME_METHOD_IMGS)) if rd else []
        print(f"[{ticket}] time methodology ({len(caps)} imgs) ...")
        m["time"] = post_comment(ticket, time_method_doc(caps))
        save_state(st)


def do_pass(log_dirs, only):
    st = load_state()
    scenarios, presets = scan(log_dirs)
    tickets = sorted({s["ticket"] for s in scenarios.values()})
    for t in tickets:
        if only and t != only:
            continue
        ensure_method(t, scenarios, presets, st)

    # group scenarios into one comment per (ticket, MTP / non-MTP)
    groups = {}
    for key in sorted(scenarios):
        s = scenarios[key]
        if only and s["ticket"] != only:
            continue
        if not (s["token"] or s["time"]):
            continue
        gkey = "nomtp" if mtp_of(s["preset"]) == "MTP 0" else "mtp"
        groups.setdefault((s["ticket"], gkey), []).append(s)

    for (ticket, gkey), members in sorted(groups.items()):
        ensure_method(ticket, scenarios, presets, st)
        members.sort(key=lambda s: (ISL_ORDER.get(s["name"].split("_")[0], 9),
                                    int(parse_scenario(s["name"])[2])
                                    if parse_scenario(s["name"])[2].isdigit() else 0))
        label = "no MTP (MTP 0)" if gkey == "nomtp" else "MTP (speculative)"
        n = len(members)
        # config 1 is the group "main" comment; configs 2..N are threaded replies
        main_key = f"{ticket}|main|{gkey}"
        main_id = st.setdefault("group_main", {}).get(main_key)
        for idx, s in enumerate(members):
            skey = f"{ticket}|cfg|{gkey}|{s['name']}"
            if st["scenarios"].get(skey):
                continue
            tok = json.load(open(s["token"]["summary"])) if s["token"] else None
            tim = json.load(open(s["time"]["summary"])) if s["time"] else None
            # Group MAIN (idx 0) carries image rows; rest are text-only — keeps total attachments
            # low (Jira drops thumbnails when an issue has too many). JIRA_IMG_ALL=1 -> every config
            # gets images (small sweeps like kimi).
            if idx == 0 or IMG_ALL:
                tok_paths = collect_paths(s["token"]["analysis"], GROUP_TOKEN_IMGS) if s["token"] else []
                tim_paths = collect_paths(s["time"]["analysis"], GROUP_TIME_IMGS) if s["time"] else []
            else:
                tok_paths = tim_paths = []
            kind = "main" if idx == 0 else f"reply->{main_id}"
            print(f"[{ticket}] {label} config {idx + 1}/{n} ({kind}): {s['name']} "
                  f"(token {len(tok_paths)} + time {len(tim_paths)} imgs) ...")
            tok_ids = upload_group(ticket, tok_paths)
            tim_ids = upload_group(ticket, tim_paths)
            adf = config_doc(label, idx == 0, n, idx, s["model"], s["preset"],
                             s["name"], tok, tim, tok_ids, tim_ids)
            cid = post_comment(ticket, adf, parent_id=None if idx == 0 else main_id)
            st["scenarios"][skey] = cid
            if idx == 0:
                main_id = cid
                st["group_main"][main_key] = cid
            save_state(st)

    # two FINAL comments per ticket: summary tables over all configs (no images)
    for t in tickets:
        if only and t != only:
            continue
        tok_doc, tim_doc = summary_table_docs(t, scenarios)
        for key, dd in (("table_token", tok_doc), ("table_time", tim_doc)):
            skey = f"{t}|{key}"
            if st["scenarios"].get(skey):
                continue
            print(f"[{t}] summary {key} ...")
            st["scenarios"][skey] = post_comment(t, dd)
            save_state(st)
    return len(scenarios)


def _patch_conclusion(body, new_text):
    """In-place: replace the 'Conclusion: ...' paragraph's text. Return True if found."""
    for node in body.get("content", []):
        if node.get("type") != "paragraph":
            continue
        kids = node.get("content", [])
        if kids and kids[0].get("type") == "text" and kids[0].get("text", "").startswith("Conclusion:"):
            node["content"] = [_text(new_text)]
            return True
    return False


def update_text(log_dirs, only):
    """Re-patch ONLY the text that depends on the MoE-compute vs comm relationship:
       the per-config 'Conclusion:' paragraph and the TIME-summary footer. Media UUIDs
       already attached are reused verbatim, so no images are re-uploaded and no orphan
       attachments are created. Use after fixing the operator/narrative logic."""
    st = load_state()
    scenarios, _ = scan(log_dirs)
    n_cfg = n_sum = 0
    for key in sorted(scenarios):
        s = scenarios[key]
        if (only and s["ticket"] != only) or not (s["token"] or s["time"]):
            continue
        gkey = "nomtp" if mtp_of(s["preset"]) == "MTP 0" else "mtp"
        skey = f"{s['ticket']}|cfg|{gkey}|{s['name']}"
        cid = st.get("scenarios", {}).get(skey)
        if not cid:
            print(f"  skip(no comment id) {skey}")
            continue
        tok = json.load(open(s["token"]["summary"])) if s["token"] else None
        tim = json.load(open(s["time"]["summary"])) if s["time"] else None
        ct = conclusion_text(tok, tim)
        if not ct:
            continue
        if DRY:
            print(f"  [DRY] {skey}\n        -> {ct}")
            n_cfg += 1
            continue
        body = get_comment_body(s["ticket"], cid)
        if _patch_conclusion(body, ct):
            put_comment(s["ticket"], cid, body)
            print(f"  patched conclusion: {s['ticket']} {gkey} {s['name']} (cid {cid})")
            n_cfg += 1
        else:
            print(f"  WARN no 'Conclusion' para in {s['ticket']} {skey} (cid {cid})")
    tickets = sorted({s["ticket"] for s in scenarios.values() if not only or s["ticket"] == only})
    for t in tickets:
        skey = f"{t}|table_time"
        cid = st.get("scenarios", {}).get(skey)
        if not cid:
            print(f"  skip(no summary id) {skey}")
            continue
        _, tim_doc = summary_table_docs(t, scenarios)
        if DRY:
            print(f"  [DRY] {t} table_time footer rebuild")
        else:
            put_comment(t, cid, tim_doc)
            print(f"  patched TIME summary footer: {t} (cid {cid})")
        n_sum += 1
    print(f"[update-text done] {n_cfg} config conclusions, {n_sum} summary footers")


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=DEFAULT_LOGS,
                    help="comma-separated run roots (logs/ dirs)")
    ap.add_argument("--only", default="", help="only this ticket, e.g. MV-4571")
    ap.add_argument("--loop", type=int, default=0, help="re-scan every N seconds (0=once)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update-text", action="store_true",
                    help="patch only the MoE-vs-comm conclusion/footer text on already-posted "
                         "comments (reuses media, no re-upload)")
    args = ap.parse_args()
    DRY = args.dry_run
    if not DRY and (not EMAIL or not TOKEN):
        sys.exit("ERROR: set JIRA_EMAIL and JIRA_API_TOKEN env (or use --dry-run).")
    log_dirs = [d.strip() for d in args.logs.split(",") if d.strip()]
    if args.update_text:
        update_text(log_dirs, args.only)
        return
    while True:
        n = do_pass(log_dirs, args.only)
        print(f"[pass done] {n} scenarios seen @ {time.strftime('%H:%M:%S')}")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
