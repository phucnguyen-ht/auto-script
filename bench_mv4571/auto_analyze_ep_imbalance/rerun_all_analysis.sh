#!/usr/bin/env bash
# =============================================================================
# rerun_all_analysis.sh — MV-4571
#   1) Gộp toàn bộ logs từ *-gpu-6 về *-gpu-5 thành MỘT folder logs/ duy nhất.
#   2) Chạy lại analyze_time.py (phase=cudagraph) + analyze_tokens.py (threshold=auto)
#      trên TẤT CẢ cấu hình đã chạy, song song, ghi đè <case>/{time,tokens}/analysis/.
#
# Mỗi (case, phase) độc lập -> chạy lại idempotent (overwrite output, không đụng raw
# serve.log / traces). Sau khi chạy xong có thể push:
#   python3 jira_push.py --only MV-4571 --logs logs
# (chỉ còn 1 root vì đã gộp).
#
# Env tuỳ chỉnh:
#   JOBS=8            số job song song (mặc định 8)
#   MERGE=1           1=gộp gpu-6 vào gpu-5 trước khi chạy (mặc định); 0=bỏ qua, chạy luôn
#   PHASE_METHOD=cudagraph   cudagraph|kernel|auto (analyze_time)
#   DECODE_THR=auto          auto|4xconc|6xconc|<số> (analyze_tokens)
#   GANTT_REPR="noMTP-bs64-dg/8k_rinf_c22"  scenario DUY NHẤT được vẽ gantt (ảnh methodology)
#   ONLY_TIME=0 / ONLY_TOKEN=0   chỉ chạy 1 loại phase
#   DRY=0             1=chỉ in lệnh, không chạy
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G5_LOGS="$HERE/logs"
ROOT="$(cd "$HERE/../../.." 2>/dev/null && pwd)"   # = .../mv-4571
G6_LOGS="$(readlink -m "$ROOT/auto-script-gpu-6/bench_mv4571/auto_analyze_ep_imbalance/logs")"

JOBS="${JOBS:-16}"
MERGE="${MERGE:-1}"
PHASE_METHOD="${PHASE_METHOD:-cudagraph}"
DECODE_THR="${DECODE_THR:-auto}"
GANTT_REPR="${GANTT_REPR:-noMTP-bs64-dg/8k_rinf_c22}"
ONLY_TIME="${ONLY_TIME:-1}"; ONLY_TOKEN="${ONLY_TOKEN:-1}"
DRY="${DRY:-0}"
PYBIN="${PYBIN:-python3}"
RELOG="$HERE/rerun_logs"; mkdir -p "$RELOG"
# log path có cấu trúc: rerun_logs/<run_ts>/<model>/<preset>/<case>/<phase>.log
logpath_for() {  # $1 = base (.../logs/run_XXX/model/preset/case/{time,tokens})
  local rel="${1#$G5_LOGS/}"; rel="${rel#run_}"   # XXX/model/preset/case/phase
  echo "$RELOG/$(dirname "$rel")/$(basename "$rel").log"
}

mkdir -p "$G5_LOGS"

# ---------- 1) MERGE gpu-6 -> gpu-5 ----------
if [ "$MERGE" = "1" ] && [ -d "$G6_LOGS" ]; then
  echo "[merge] $G6_LOGS  ->  $G5_LOGS"
  for run in "$G6_LOGS"/run_*; do
    [ -d "$run" ] || continue
    dest="$G5_LOGS/$(basename "$run")"
    echo "        + $(basename "$run")"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --ignore-existing "$run/" "$dest/"
    else
      mkdir -p "$dest"; cp -rn "$run/." "$dest/" 2>/dev/null || true
    fi
  done
else
  echo "[merge] skipped (MERGE=$MERGE)"
fi

# ---------- 2) build job list ----------
JOBS_FILE="$(mktemp)"
: > "$JOBS_FILE"

if [ "$ONLY_TIME" = "1" ]; then
  while IFS= read -r tdir; do
    base="$(dirname "$tdir")"                       # .../<case>/time
    rel="$(basename "$(dirname "$base")")/$(basename "$(dirname "$tdir")")"  # preset/<...> (best-effort)
    # gantt chỉ cho scenario đại diện
    gantt="--no-gantt"
    case "$base" in *"$GANTT_REPR"/time) gantt="";; esac
    logf="$(logpath_for "$base")"; mkdir -p "$(dirname "$logf")"
    printf '%s -u "%s/analyze_time.py" --trace-dir "%s" --out "%s/analysis" --phase-method %s --no-verify-schema %s > "%s" 2>&1\n' \
      "$PYBIN" "$HERE" "$tdir" "$base" "$PHASE_METHOD" "$gantt" "$logf" >> "$JOBS_FILE"
  done < <(find "$G5_LOGS" -type d -path "*/time/traces" | sort)
fi

if [ "$ONLY_TOKEN" = "1" ]; then
  while IFS= read -r sl; do
    base="$(dirname "$sl")"                          # .../<case>/tokens
    conc="$(basename "$(dirname "$base")" | sed -n 's/.*_c\([0-9][0-9]*\).*/\1/p')"; conc="${conc:-8}"
    logf="$(logpath_for "$base")"; mkdir -p "$(dirname "$logf")"
    printf '%s -u "%s/analyze_tokens.py" --log "%s" --out "%s/analysis" --decode-max-tokens %s --concurrency %s > "%s" 2>&1\n' \
      "$PYBIN" "$HERE" "$sl" "$base" "$DECODE_THR" "$conc" "$logf" >> "$JOBS_FILE"
  done < <(find "$G5_LOGS" -type f -path "*/tokens/serve.log" | sort)
fi

NJOB="$(wc -l < "$JOBS_FILE")"
echo "[jobs] $NJOB (JOBS=$JOBS, phase=$PHASE_METHOD, decode-thr=$DECODE_THR)"
if [ "$DRY" = "1" ]; then echo "[dry] commands:"; cat "$JOBS_FILE"; rm -f "$JOBS_FILE"; exit 0; fi
[ "$NJOB" -eq 0 ] && { echo "[warn] không tìm thấy case nào"; rm -f "$JOBS_FILE"; exit 0; }

# ---------- 3) run parallel (semaphore JOBS) ----------
ok=0; fail=0; done=0
run_one() { bash -c "$1"; }
while IFS= read -r cmd; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || wait; done
  { run_one "$cmd" && echo "OK" >>"$JOBS_FILE.res" || echo "FAIL" >>"$JOBS_FILE.res"; } &
done < "$JOBS_FILE"
wait

ok="$(grep -c OK "$JOBS_FILE.res" 2>/dev/null || echo 0)"
fail="$(grep -c FAIL "$JOBS_FILE.res" 2>/dev/null || echo 0)"
echo "[done] ok=$ok fail=$fail  (log per-case: $RELOG/)"
[ "$fail" -gt 0 ] && echo "[note] xem *.log trong $RELOG để biết case fail (thường do serve.log không có [EP_COLLECT])."
rm -f "$JOBS_FILE" "$JOBS_FILE.res"
echo "[next] push: cd $HERE && JIRA_EMAIL=... JIRA_API_TOKEN=... python3 jira_push.py --only MV-4571 --logs logs"
