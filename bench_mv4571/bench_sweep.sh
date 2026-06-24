#!/usr/bin/env bash
# MV-4524: GLM-5.1-FP8 1P1D (MI300X, DP8-EP8) benchmark sweep over LongBench-v2.
#
# Ticket requirement: concurrency {8 16 24 32 40 48 56 64}, DP8-EP8, dataset
# /remote/vast0/share-mv/longbenchv2-custom. Results -> Google Sheet (Anh Cao analyzes).
#
# ISL/OSL per case (per duc-dam):  8k -> 8192/1024 ,  10k -> 10000/500 ,  100k -> 100000/500.
#   (ISL is set by the dataset prompt length; OSL via --custom-output-len.)
# Request rates (per duc-dam): inf FIRST, then 0.25, 0.5, 1 (req/s). Each rate runs the
#   full ISL x conc sweep before the next rate.
#     * inf  = closed-loop: client holds exactly max-concurrency requests in flight.
#     * 0.25/0.5/1 = open-loop Poisson arrivals at that req/s, capped by max-concurrency.
# num-prompts (per duc-dam): 2 x concurrency  (capped at the file's unique-prompt count).
# Protocol (per ticket):
#   * RESET the prefix cache AFTER measuring each conc -> every conc measured on a cold cache.
#   * NO warmup: vanilla `vllm bench serve` (num_warmups=0); we do NOT pass --num-warmups.
#
# The benchmark is a pure HTTP client hitting the heimdall/istio GATEWAY (routes to the
# prefill mi300-7 / decode mi300-9 InferenceServices).
#
#   PREREQ 1 (server): GLM-5.1-FP8 1P1D deployed AND returning coherent output.
#   PREREQ 2 (gateway):  kubectl -n mv-ducdam-4524 port-forward svc/mif-istio 18080:80
#   PREREQ 3 (reset, REQUIRED): port-forward the PREFILL pod, set RESET_URL
#       kubectl -n mv-ducdam-4524 port-forward <prefill-pod> 8001:8000 ; export RESET_URL=http://localhost:8001
set -euo pipefail

# ---- target (must match served-model-name in inf_1p1d_*.yaml) ---------------
MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
TOKENIZER="${TOKENIZER:-/remote/vast0/duc-dam/MV-4524/glm52-tok}"   # patched GLM-5.2 tokenizer; exact ISL
BASE_URL="${BASE_URL:-http://localhost:18080}"                    # gateway port-forward
ENDPOINT="${ENDPOINT:-/v1/completions}"                           # + --skip-chat-template => prompt_len == ISL

DATA="${DATA:-/remote/vast0/duc-dam/MV-4524/longbenchv2-custom}"

# ---- sweep params (MV-4524) -------------------------------------------------
ISLS="${ISLS:-8k 10k 100k 1M}"            # 8k -> 10k -> 100k -> 1M (GLM-5.2 supports 1M ctx)
CONCS="${CONCS:-8 22 31 36 40 52 64}"  # MV-4524 GLM-5.2 set (8k/10k)
CONCS_100K="${CONCS_100K:-8 22 31 36}" # MV-4524 split bench: 100k CHỈ đo đến conc 36
CONCS_1M="${CONCS_1M:-1 2 4 8}"        # MV-4524 GLM-5.2: 1M dùng conc thấp riêng (prefill 1M quá nặng ở conc cao)
REQUEST_RATES="${REQUEST_RATES:-inf 0.25 0.5 1}"   # inf FIRST, then 0.25, 0.5, 1 (req/s)
TAG="${TAG:-dp8ep8}"                       # label for files/log
RESET_URL="${RESET_URL:-}"                 # engine URL(s) for /reset_prefix_cache (space-separated). REQUIRED.
ROOT_OUT="${ROOT_OUT:-results}"
# NUM_PROMPTS (optional): if set, fixes num-prompts for ALL concs; else default = 2 x conc.

# per dataset label -> "ISL_TOKENS  OSL  CAP(unique prompts in file)"
osl_cap_for() {
  case "$1" in
    8k)   echo "8192 1024 256"  ;;
    10k)  echo "10000 500 256"  ;;
    100k) echo "100000 500 100" ;;
    1M)   echo "1000000 500 22" ;;
    *)    echo "ERR ERR ERR"    ;;
  esac
}

reset_cache() {  # reset prefix cache on every RESET_URL (one API server resets all DP ranks)
    if [ -z "$RESET_URL" ]; then
        echo "  !! RESET_URL unset -> prefix cache NOT reset. Ticket requires a reset after each conc."
        return
    fi
    local u code
    for u in $RESET_URL; do
        code=$(curl -X POST -s -o /dev/null -w "%{http_code}" "$u/reset_prefix_cache" || echo "000")
        echo "  reset_prefix_cache($u) -> HTTP $code"
        [ "$code" = "200" ] || echo "    WARNING: reset failed at $u (wrong URL? dev mode off?)"
    done
}

summarize() {  # $1=ISL(label)  $2=ISLTOK  $3=RATELABEL  $4=OUTDIR
    local ISL="$1" ISLTOK="$2" RATELABEL="$3" OUTDIR="$4"
    ISL="$ISL" ISLTOK="$ISLTOK" RATELABEL="$RATELABEL" CONCS="$CONCS" OUTDIR="$OUTDIR" TAG="$TAG" python3 - <<'PY'
import json, os
isl=os.environ["ISL"]; isltok=os.environ["ISLTOK"]; rl=os.environ["RATELABEL"]; tag=os.environ["TAG"]; outdir=os.environ["OUTDIR"]
concs=[int(x) for x in os.environ["CONCS"].split()]
def g(d,*ks):
    for k in ks:
        if d.get(k) is not None: return d[k]
    return None
hdr=f"{'conc':>5} {'nprompt':>7} | {'req/s':>7} {'out tok/s':>9} | {'TTFT mean':>9} {'TTFT p99':>9} | {'TPOT mean':>9} {'TPOT p99':>9} | {'ITL mean':>8} | {'dur s':>7}"
lines=[f"# ISL={isltok} ({isl}) rate={rl} tag={tag}  (num-prompts=2xconc; reset after each conc; no warmup)", hdr, "-"*len(hdr)]
for c in concs:
    p=f"{outdir}/glm51_{isl}_c{c}_{rl}_{tag}.json"
    if not os.path.exists(p):
        lines.append(f"{c:>5} {'-':>7} | {'(missing)':>7}"); continue
    d=json.load(open(p))
    def f(*ks,fmt="{:.1f}"):
        v=g(d,*ks); return fmt.format(v) if v is not None else "-"
    np=g(d,"num_prompts","completed"); np = int(np) if np is not None else "-"
    lines.append(f"{c:>5} {str(np):>7} | {f('request_throughput'):>7} {f('output_throughput'):>9} | "
                 f"{f('mean_ttft_ms'):>9} {f('p99_ttft_ms'):>9} | "
                 f"{f('mean_tpot_ms'):>9} {f('p99_tpot_ms'):>9} | "
                 f"{f('mean_itl_ms'):>8} | {f('duration'):>7}")
table="\n".join(lines)
print(table)
open(f"{outdir}/summary_{tag}_{rl}_{isl}.txt","w").write(table+"\n")
print(f"[saved -> {outdir}/summary_{tag}_{rl}_{isl}.txt]")
PY
}

run_isl() {  # $1=ISL(label)  $2=RATE
    local ISL="$1" RATE="$2"; local ISLTOK OSL CAP NP OUTDIR RATELABEL
    read -r ISLTOK OSL CAP < <(osl_cap_for "$ISL")
    [ "$ISLTOK" = "ERR" ] && { echo "!! unknown ISL '$ISL' (use 8k|10k|100k) — skipping"; return; }
    RATELABEL="r${RATE}"
    OUTDIR="$ROOT_OUT/${TAG}_${RATELABEL}_${ISL}"; mkdir -p "$OUTDIR"
    local CLIST="$CONCS"; [ "$ISL" = "100k" ] && CLIST="$CONCS_100K"; [ "$ISL" = "1M" ] && CLIST="$CONCS_1M"   # MV-4524 split: 100k->conc<=36, 1M->1 2 4 8
    echo ""; echo "########## rate=$RATE  ISL=$ISL (${ISLTOK} tok)  OSL=$OSL  num-prompts=2xconc(no cap)  tag=$TAG  concs: $CLIST ##########"
    echo "--- initial reset (so conc 1 also starts cold) ---"; reset_cache
    for C in $CLIST; do
        local RF="$OUTDIR/glm51_${ISL}_c${C}_${RATELABEL}_${TAG}.json"
        [ -f "$RF" ] && { echo "--- SKIP (already done): rate=$RATE ISL=$ISL conc=$C ---"; continue; }
        NP="${NUM_PROMPTS:-$(( 2 * C ))}"           # MV-4524 GLM-5.2: num-prompts = 2x conc, NO cap (prefix cache disabled -> prompt repeats are fine)
        echo "--- rate=$RATE ISL=$ISL conc=$C num-prompts=$NP ---"
        vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" \
            --trust-remote-code \
            --dataset-name custom \
            --dataset-path "$DATA/longbenchv2-$ISL.jsonl" \
            --skip-chat-template \
            --custom-output-len "$OSL" \
            --ignore-eos \
            --base-url "$BASE_URL" --endpoint "$ENDPOINT" \
            --num-prompts "$NP" --max-concurrency "$C" \
            --request-rate "$RATE" \
            --percentile-metrics ttft,tpot,e2el,itl \
            --metric-percentiles 75,90,99 \
            --save-result \
            --metadata model="$MODEL" isl="$ISLTOK" osl="$OSL" conc="$C" num_prompts="$NP" request_rate="$RATE" tag="$TAG" \
            --result-filename "$OUTDIR/glm51_${ISL}_c${C}_${RATELABEL}_${TAG}.json"
        echo "--- reset prefix cache AFTER conc=$C (ticket protocol) ---"; reset_cache
    done
    echo "=== summary rate=$RATE ISL=$ISL ==="; summarize "$ISL" "$ISLTOK" "$RATELABEL" "$OUTDIR"
}

mkdir -p "$ROOT_OUT"
exec > >(tee "$ROOT_OUT/bench-${TAG}-all.log") 2>&1
echo "=== MV-4524 sweep | model=$MODEL | rates (in order): $REQUEST_RATES | ISLs: $ISLS | concs: $CONCS | num-prompts=2xconc | tag=$TAG ==="
echo "=== url=$BASE_URL$ENDPOINT | reset=${RESET_URL:-NONE!} | tokenizer=$TOKENIZER | warmup=0 ==="
for RATE in $REQUEST_RATES; do
    echo ""; echo "==================== REQUEST RATE = $RATE ===================="
    for ISL in $ISLS; do
        run_isl "$ISL" "$RATE"
    done
done
echo ""; echo "=== DONE. Summaries: $ROOT_OUT/${TAG}_r<RATE>_<ISL>/summary_${TAG}_r<RATE>_<ISL>.txt -> paste into the sheet ==="
