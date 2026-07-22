#!/usr/bin/env bash
# Short EPLB smoke test in the freshly-built retest container (vllm-moreh from retest source
# WITH patch 12). Resilient to GPU contention: wait while a foreign container holds the GPU,
# serve ONE EPLB preset (1 conc), confirm (a) EPLB actually ENGAGES (nixl eplb-* agents — only
# if supports_eplb from the patch works), (b) it serves + generates. Writes VERDICT.
set -uo pipefail
CN="${CN:-phuc-nguyen-mv4572-rebench-retest}"
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
MODEL=/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/
PRESET="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml"
AITER="$WD/retest/vllm-moreh/src/aiter_moreh"
OUT="$WD/auto-script/bench_mv4572/logs/retest_eplb_test"; mkdir -p "$OUT"
SERVE="$OUT/serve.log"; VERD="$OUT/VERDICT.txt"
POLICE=/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh
foreign() { bash "$POLICE" 2>/dev/null | awk '/^[[:space:]]+[^[:space:]].*id=/{print $1}' | grep -vx "$CN" || true; }
running() { [ "$(podman inspect -f '{{.State.Running}}' "$CN" 2>/dev/null)" = "true" ]; }
try=0; MAX="${MAX:-40}"
while [ "$try" -lt "$MAX" ]; do
  try=$((try+1))
  while [ -n "$(foreign)" ]; do echo "[retest-eplb] GPU busy by $(foreign|paste -sd,) -> wait 60s"; sleep 60; done
  running || { echo "[retest-eplb] podman restart $CN"; podman restart "$CN" >/dev/null 2>&1; sleep 5; }
  podman exec "$CN" bash -lc 'pkill -9 VLLM 2>/dev/null; true' >/dev/null 2>&1; sleep 2
  echo "[retest-eplb] attempt $try: serve $(basename "$PRESET")"; : > "$SERVE"
  podman exec -d "$CN" bash -lc "export AITER_MOREH_ROOT_DIR='$AITER'; vllm-moreh serve '$MODEL' '$PRESET' > '$SERVE' 2>&1"
  t=0; ok=0
  while [ "$t" -lt 1200 ]; do
    podman exec "$CN" bash -lc 'curl -sf http://localhost:8000/health >/dev/null 2>&1' && { ok=1; break; }
    running || { echo "[retest-eplb] container down mid-serve -> retry"; break; }
    sleep 15; t=$((t+15))
  done
  [ "$ok" = 1 ] || { echo "[retest-eplb] not healthy (attempt $try)"; podman exec "$CN" bash -lc 'pkill -9 VLLM 2>/dev/null; true' >/dev/null 2>&1; sleep 15; continue; }
  echo "[retest-eplb] HEALTH OK"; sleep 8
  # short 1-conc generation (5 requests) to confirm it serves + generates
  gen_ok=0
  for i in 1 2 3 4 5; do
    r=$(podman exec "$CN" bash -lc "curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":16,\"temperature\":0}'" 2>/dev/null)
    echo "$r" | grep -q '"choices"' && gen_ok=$((gen_ok+1))
  done
  eplb_agents=$(grep -acE 'Initialized NIXL agent: eplb-' "$SERVE" 2>/dev/null)
  eplb_mark=$(grep -acE 'EPLB rank|Rearranging experts|eplb_state|enable_eplb' "$SERVE" 2>/dev/null)
  disabled=$(grep -acE 'supports_eplb.*[Ff]alse|EPLB.*disabled|does not support eplb' "$SERVE" 2>/dev/null)
  {
    echo "RETEST EPLB SMOKE TEST ($(date))"
    echo "container=$CN  vllm=retest-source+patch12  preset=$(basename "$PRESET")"
    echo "nixl eplb agents initialized : $eplb_agents  (expect ~8 => EPLB engaged via supports_eplb)"
    echo "eplb engage markers          : $eplb_mark"
    echo "eplb-disabled markers        : $disabled  (expect 0)"
    echo "generation 200/choices       : $gen_ok/5"
    if [ "${eplb_agents:-0}" -ge 1 ] && [ "${disabled:-0}" -eq 0 ] && [ "${gen_ok:-0}" -ge 1 ]; then
      echo "RESULT: PASS -> patch chuẩn: EPLB engaged (nixl) + serve + generate OK."
    else
      echo "RESULT: CHECK -> xem $SERVE"
    fi
  } | tee "$VERD"
  podman exec "$CN" bash -lc 'pkill -9 VLLM 2>/dev/null; true' >/dev/null 2>&1
  exit 0
done
echo "[retest-eplb] gave up after $MAX attempts" | tee "$VERD"