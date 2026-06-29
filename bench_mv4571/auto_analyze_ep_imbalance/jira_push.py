#!/usr/bin/env python3
# =============================================================================
# MV-4571/4572 — push EP-imbalance sweep results to Jira (loop + idempotent).
#
# Watches bench_mv4571/auto_analyze_ep_imbalance/logs/run_*/ , and for each model
# ticket posts:
#   1) one TOKEN methodology comment (how token-imbalance is computed + the
#      analyze_tokens.py script + what its main blocks do),
#   2) one TIME  methodology comment (MoE comm->compute->comm barrier model +
#      the analyze_time.py script + verify/analyze blocks),
#   3) one comment PER completed scenario: numbers -> images -> conclusion.
#
# Idempotent: a JSON state file records what was already posted, so re-running
# (or the --loop) only posts NEW things. Images are uploaded as issue
# attachments (REST) and embedded in the comment via ADF media nodes.
#
# Auth (Jira Cloud): export JIRA_EMAIL + JIRA_API_TOKEN (create at
#   https://id.atlassian.com/manage-profile/security/api-tokens ). Uses Basic auth.
#
# Usage:
#   JIRA_EMAIL=you@moreh.com.vn JIRA_API_TOKEN=xxxx \
#     python3 jira_push.py --once          # one pass over current logs
#   ... --loop 120                          # re-scan every 120s (push as sweep finishes)
#   ... --dry-run                           # build everything, post nothing
#   [--only MV-4571]  [--logs <dir>]
# =============================================================================
import argparse
import json
import os
import subprocess
import sys
import time

SITE = "https://moreh.atlassian.net"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGS = os.path.join(SCRIPT_DIR, "logs")
STATE_FILE = os.path.join(SCRIPT_DIR, ".jira_push_state.json")
TOKENS_PY = os.path.join(SCRIPT_DIR, "analyze_tokens.py")
TIME_PY = os.path.join(SCRIPT_DIR, "analyze_time.py")

MODEL_TICKET = {"glm5.2": "MV-4571", "kimi2.6": "MV-4572"}

# images embedded per scenario (only those that exist are attached), in order.
TOKEN_IMGS = ["token_imbalance_hist_decode.png", "imbalance_per_layer.png",
              "load_heatmap_layer_x_rank.png"]
TIME_IMGS = ["per_rank_time_breakdown.png", "time_imbalance_hist_decode.png"]

EMAIL = os.environ.get("JIRA_EMAIL", "")
TOKEN = os.environ.get("JIRA_API_TOKEN", "")
DRY = False


# ----------------------------- REST (via curl) -----------------------------
def _curl(args):
    r = subprocess.run(["curl", "-sS", "--fail-with-body", "-u", f"{EMAIL}:{TOKEN}"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr or r.stdout}")
    return r.stdout


def upload_attachment(key, path):
    """Attach a file to the issue; return its attachment id (for ADF media)."""
    out = _curl(["-X", "POST", "-H", "X-Atlassian-Token: no-check",
                 "-F", f"file=@{path}", f"{SITE}/rest/api/3/issue/{key}/attachments"])
    return json.loads(out)[0]["id"]


def post_comment(key, adf):
    body = json.dumps({"body": adf})
    out = _curl(["-X", "POST", "-H", "Content-Type: application/json",
                 "--data", body, f"{SITE}/rest/api/3/issue/{key}/comment"])
    return json.loads(out)["id"]


# ----------------------------- ADF builders -----------------------------
def _text(s):
    return {"type": "text", "text": s}


def heading(s, level=3):
    return {"type": "heading", "attrs": {"level": level}, "content": [_text(s)]}


def para(s):
    return {"type": "paragraph", "content": ([_text(s)] if s else [])}


def code(s, lang="python"):
    return {"type": "codeBlock", "attrs": {"language": lang}, "content": ([_text(s)] if s else [])}


def bullets(items):
    return {"type": "bulletList",
            "content": [{"type": "listItem", "content": [para(i)]} for i in items]}


def media(att_id):
    return {"type": "mediaSingle", "attrs": {"layout": "center"},
            "content": [{"type": "media", "attrs": {"id": att_id, "type": "file", "collection": ""}}]}


def doc(blocks):
    return {"type": "doc", "version": 1, "content": blocks}


# ----------------------------- methodology content -----------------------------
def read_script(path, max_chars=28000):
    try:
        s = open(path).read()
    except OSError:
        return "(script not found)"
    return s if len(s) <= max_chars else s[:max_chars] + "\n# ... (truncated; see repo) ...\n"


def token_method_doc():
    b = [
        heading("EP imbalance by #tokens — token-routing load before all2all", 2),
        para("On DP8/EP8, before the all-gather dispatch of each MoE (layer, step), every rank "
             "logs the histogram of its tokens' top-8 expert ids over all 256 experts "
             "([EP_COLLECT], gated by VLLM_MOREH_EP_LOG=1, requires enforce_eager)."),
        bullets([
            "Aggregate: sum the 8 ranks' histograms per (layer, step) -> global per-expert "
            "routing count g[256].",
            "Linear placement: rank r owns experts [32r, 32r+32). load[r] = sum of g over r's "
            "32 experts = #token-routings rank r must compute.",
            "imbalance per (layer, step) = max_r(load) / min_r(load).",
            "phase split: decode if num_tokens <= 4*concurrency else prefill_mixed "
            "(num_tokens = total_routings / topk).",
            "token-ratio is a proxy: it overstates the real cost (tiling/padding + fixed "
            "overhead) -> the TIME analysis is the ground truth.",
        ]),
        heading("Script: analyze_tokens.py", 3),
        para("Main blocks: parse_and_aggregate (regex the [EP_COLLECT] lines, normalize step = "
             "it - min_it per (rank,layer), sum counts -> global g per (layer,step)); "
             "build_steps_df + rank_loads (split g into 8 contiguous expert-blocks, "
             "compute max/min, max/avg per step; auto phase by 4*concurrency); "
             "plot_imbalance_hist (max/min histograms all/prefill/decode); "
             "plot_load_distribution + plot_per_layer_load (per-expert / per-rank load, "
             "layer x {expert,rank} heatmaps); plot_per_layer_imbalance (mean imbalance per "
             "layer + decode layer x step heatmap); export (steps_imbalance.parquet, "
             "expert_dist_global.npy, summary_tokens.json)."),
        code(read_script(TOKENS_PY)),
    ]
    return doc(b)


def time_method_doc():
    b = [
        heading("EP imbalance by TIME — MoE comm -> compute -> comm (barrier model)", 2),
        para("MoE pattern on DP8/EP8: communication (all-gather) -> MoE compute -> "
             "communication (reduce-scatter). Each (layer, step) forms one such cluster on every "
             "rank; the 8 ranks run lockstep, so the k-th cluster on all 8 ranks is the same "
             "(layer, step). We map clusters by order of appearance (cluster k of rank0 <-> "
             "cluster k of rank1 ... rank7)."),
        para("Expected (barrier) behavior: all-gather and reduce-scatter are synchronous "
             "collectives, so per cluster the gather-END of the 8 ranks ~coincide (one barrier -> "
             "every rank's MoE starts together), and the reduce-scatter-END of the 8 ranks "
             "~coincide (one barrier). Only the MoE compute in between differs per rank (different "
             "load) -> a rank with longer MoE starts reduce-scatter later, but all finish "
             "reduce-scatter together (fast ranks wait for the slowest)."),
        para("Per mapped cluster we have, for all 8 ranks: time(start,end,dur) of comm_gather; "
             "time(start,end,dur) of comm_reduce_scatter; moe_time = reduce_scatter.start - "
             "gather.end."),
        para("Phase by kernel chain: decode = moe_sorting -> quant -> moe_stage_1 -> quant -> "
             "moe_stage_2 ; mixed-prefill = moe_sorting -> quant -> moe_fused."),
        para("imbalance per cluster: imbalance_time = max_moe_time(8 ranks) / min_moe_time(8 ranks). "
             "Headroom = 1 - sum(mean_r moe)/sum(max_r moe) (critical path is sum of per-step max "
             "because reduce-scatter is a barrier)."),
        heading("Script: analyze_time.py", 3),
        para("Main blocks: verify_schema (CHECK1-4: trace header/kernel names, shared clock across "
             "ranks, #comm equal across 8 ranks = mapping valid); parse_all (comm-bounded parse: "
             "open a cluster by sort/gemm/fmoe NOT by quant, gather vs reduce-scatter by POSITION, "
             "moe = rs.start - gather.end, nstage 1=fused->prefill / 2=gemm->decode); "
             "verify_same_kernel + verify_phase (consistency); per_rank_table (MoE/gather/rscat ms "
             "per rank + breakdown.png); compute_metrics (critical=Sum max, balanced=Sum mean, "
             "headroom, max/min); plot_time_imbalance_hist (all/prefill/decode); plot_gantt "
             "(gather|moe|reduce-scatter with the two barrier lines); verify_barrier_rel/abs; "
             "export (trace_pairs.npz, summary_time.json)."),
        code(read_script(TIME_PY)),
    ]
    return doc(b)


# ----------------------------- per-scenario content -----------------------------
def mtp_of(preset):
    p = preset.lower()
    if "nomtp" in p:
        return "MTP 0"
    for n in ("3", "5"):
        if f"mtp{n}" in p:
            return f"MTP {n}"
    return preset


def parse_scenario(name):  # e.g. 8k_rinf_c8
    isl = name.split("_")[0]
    conc = name.split("_c")[-1] if "_c" in name else "?"
    rate = "inf"
    if "_r" in name:
        rate = name.split("_r")[1].split("_")[0].replace("p", ".")
    return isl, rate, conc


def fnum(x, n=2):
    return f"{float(x):.{n}f}"


def token_scenario_doc(model, preset, name, summary, att_ids):
    isl, rate, conc = parse_scenario(name)
    dec, pre = summary.get("decode", {}), summary.get("prefill_mixed", {})
    b = [
        heading(f"[TOKEN] {model} {mtp_of(preset)} | ISL {isl} | conc {conc} | rate {rate}", 3),
        bullets([
            f"ranks={summary.get('ranks')} experts={summary.get('experts')} "
            f"topk={summary.get('topk')} experts/rank={summary.get('experts_per_rank')} "
            f"steps={summary.get('n_steps')}",
            f"decode imbalance max/min: mean {fnum(dec.get('mean',0))}, median "
            f"{fnum(dec.get('median',0))}, p99 {fnum(dec.get('p99',0))} (n={dec.get('n')})",
            f"prefill_mixed imbalance max/min: mean {fnum(pre.get('mean',0))}, p99 "
            f"{fnum(pre.get('p99',0))} (n={pre.get('n')})",
            f"cumulative per-rank load max/min = {fnum(summary.get('e2e_rank_max_over_min',0))} "
            f"(balanced over the whole run).",
        ]),
    ]
    b += [media(a) for a in att_ids]
    dm = float(dec.get("mean", 0) or 0)
    b.append(para(
        f"Conclusion: per-step decode token-routing imbalance is ~{fnum(dm)} (max/min), while the "
        f"cumulative per-rank load is ~{fnum(summary.get('e2e_rank_max_over_min',0))} (i.e. balanced "
        f"on aggregate but skewed step-by-step). Token-ratio overstates the real impact; see the "
        f"[TIME] comment for the actual compute headroom."))
    return doc(b)


def time_scenario_doc(model, preset, name, summary, att_ids):
    isl, rate, conc = parse_scenario(name)
    dec = summary.get("decode", {})
    hp = float(summary.get("headroom_pct", 0) or 0)
    mm = float(dec.get("maxmin_mean", 0) or 0)
    crit = float(summary.get("critical_path_ms", 0) or 0)
    bal = float(summary.get("balanced_ms", 0) or 0)
    comm = float(summary.get("comm_ms", 0) or 0)
    b = [
        heading(f"[TIME] {model} {mtp_of(preset)} | ISL {isl} | conc {conc} | rate {rate}", 3),
        bullets([
            f"clusters={summary.get('clusters')} "
            f"(prefill {summary.get('phase_by_kernel',{}).get('prefill(1-kernel/fmoe)','?')} / "
            f"decode {summary.get('phase_by_kernel',{}).get('decode(2-kernel/gemm)','?')}), "
            f"verify_same_kernel={summary.get('verify_same_kernel_pct')}%",
            f"decode MoE imbalance max/min: mean {fnum(mm)}, p99 {fnum(dec.get('maxmin_p99',0))}",
            f"critical-path (Sum max) = {crit:.0f} ms | balanced (Sum mean) = {bal:.0f} ms | "
            f"comm = {comm:.0f} ms",
            f"MoE-compute headroom if balanced = {fnum(hp,1)} %",
        ]),
    ]
    b += [media(a) for a in att_ids]
    mean_est = mm / 1.0
    impr = (1 - 1.0 / mm) * 100 if mm > 0 else 0
    b.append(para(
        f"Conclusion: per-step MoE imbalance ~{fnum(mm)} means the busiest rank is ~{fnum(mm)}x the "
        f"idlest. Since reduce-scatter is a barrier, the step is paced by the slowest rank; assuming "
        f"total MoE-compute is conserved and redistributed to the mean, the critical path Sum(max) "
        f"shrinks to Sum(mean) -> ~{fnum(hp,1)}% MoE-compute headroom. Note MoE-compute ({bal:.0f} ms) "
        f"<< communication ({comm:.0f} ms), so end-to-end gain is smaller."))
    return doc(b)


# ----------------------------- scan + state -----------------------------
def scan(logs_dir):
    """Yield dicts for each completed (scenario, phase) analysis found under logs/."""
    found = []
    if not os.path.isdir(logs_dir):
        return found
    for run in sorted(os.listdir(logs_dir)):
        runp = os.path.join(logs_dir, run)
        if not os.path.isdir(runp):
            continue
        for model in os.listdir(runp):
            if model not in MODEL_TICKET:
                continue
            mp = os.path.join(runp, model)
            for preset in os.listdir(mp):
                pp = os.path.join(mp, preset)
                if not os.path.isdir(pp):
                    continue
                for name in os.listdir(pp):
                    base = os.path.join(pp, name)
                    for phase, summ in (("token", "tokens/analysis/summary_tokens.json"),
                                        ("time", "time/analysis/summary_time.json")):
                        sp = os.path.join(base, summ)
                        if os.path.isfile(sp):
                            found.append({"ticket": MODEL_TICKET[model], "model": model,
                                          "preset": preset, "name": name, "phase": phase,
                                          "summary": sp,
                                          "analysis": os.path.dirname(sp), "run": run})
    return found


def load_state():
    if os.path.isfile(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"method": {}, "scenarios": {}}


def save_state(st):
    json.dump(st, open(STATE_FILE, "w"), indent=2)


def imgs_for(analysis_dir, phase):
    names = TOKEN_IMGS if phase == "token" else TIME_IMGS
    out = [os.path.join(analysis_dir, n) for n in names]
    out = [p for p in out if os.path.isfile(p)]
    if phase == "time":  # add one gantt sample if present
        g = sorted(__import__("glob").glob(os.path.join(analysis_dir, "gantt", "gantt_*.png")))
        if g:
            out.append(g[0])
    return out


# ----------------------------- main pass -----------------------------
def ensure_method(ticket, st):
    st.setdefault("method", {}).setdefault(ticket, {})
    for kind, builder in (("token", token_method_doc), ("time", time_method_doc)):
        if st["method"][ticket].get(kind):
            continue
        print(f"[{ticket}] posting {kind} methodology comment ...")
        if DRY:
            st["method"][ticket][kind] = "DRY"
            continue
        cid = post_comment(ticket, builder())
        st["method"][ticket][kind] = cid
        save_state(st)


def do_pass(logs_dir, only):
    st = load_state()
    items = scan(logs_dir)
    tickets = sorted({it["ticket"] for it in items})
    for t in tickets:
        if only and t != only:
            continue
        ensure_method(t, st)
    for it in items:
        if only and it["ticket"] != only:
            continue
        skey = f"{it['ticket']}|{it['model']}/{it['preset']}/{it['name']}|{it['phase']}"
        if st["scenarios"].get(skey):
            continue
        ensure_method(it["ticket"], st)  # safety: methodology before scenarios
        summary = json.load(open(it["summary"]))
        imgs = imgs_for(it["analysis"], it["phase"])
        print(f"[{it['ticket']}] posting {it['phase']} scenario {it['model']}/{it['preset']}/"
              f"{it['name']} ({len(imgs)} imgs) ...")
        if DRY:
            st["scenarios"][skey] = "DRY"
            save_state(st)
            continue
        att = [upload_attachment(it["ticket"], p) for p in imgs]
        if it["phase"] == "token":
            adf = token_scenario_doc(it["model"], it["preset"], it["name"], summary, att)
        else:
            adf = time_scenario_doc(it["model"], it["preset"], it["name"], summary, att)
        cid = post_comment(it["ticket"], adf)
        st["scenarios"][skey] = cid
        save_state(st)
    return len(items)


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--only", default="", help="only this ticket, e.g. MV-4571")
    ap.add_argument("--loop", type=int, default=0, help="re-scan every N seconds (0=once)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    DRY = args.dry_run
    if not DRY and (not EMAIL or not TOKEN):
        sys.exit("ERROR: set JIRA_EMAIL and JIRA_API_TOKEN env (or use --dry-run).")
    while True:
        n = do_pass(args.logs, args.only)
        print(f"[pass done] {n} analyses seen @ {time.strftime('%H:%M:%S')}")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
