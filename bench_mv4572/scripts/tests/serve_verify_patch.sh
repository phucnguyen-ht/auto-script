#!/usr/bin/env bash
# Verify the EPLB-enable patch in the FRESH patchtest container (image + patch applied to
# its dist-packages vllm). Resilient to GPU contention: politely wait while a foreign
# container holds the GPU, podman-restart patchtest if down, serve ONE EPLB preset, and on
# health confirm EPLB actually ENGAGED (only possible if supports_eplb=True from the patch:
# nixl 'eplb-*' agents + EPLB rank assignment appear at startup). Writes VERDICT.
set -uo pipefail
CN="${CN:-phuc-nguyen-mv4572-patchtest}"
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
MODEL=/remote/vast0/share-mv/amd/Kimi-K2.6-MXFP4/
PRESET="$WD/auto-script/presets/kimi2.6.mxfp4/dp8ep8/base-eplb-nixl-async-default-r0.yaml"
OUT="$WD/auto-script/bench_mv4572/logs/patch_verify"; mkdir -p "$OUT"
SERVE="$OUT/serve.log"; VERD="$OUT/VERDICT.txt"
POLICE=/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh
AITER=$WD/vllm-moreh/src/aiter_moreh
foreign() { bash "$POLICE" 2>/dev/null | awk '/^[[:space:]]+[^[:space:]].*id=/{print $1}' | grep -vx "$CN" || true; }
running() { [ "$(podman inspect -f '{{.State.Running}}' "$CN" 2>/dev/null)" = "true" ]; }
try=0; MAX="${MAX:-30}"
while [ "$try" -lt "$MAX" ]; do
  try=$((try+1))
  while [ -n "$(foreign)" ]; do echo "[verify] GPU busy by $(foreign|paste -sd,) -> wait 60s"; sleep 60; done
  running || { echo "[verify] podman restart $CN"; podman restart "$CN" >/dev/null 2>&1; sleep 5; }
  podman exec "$CN" bash -lc "pkill -9 VLLM 2>/dev/null; true" >/dev/null 2>&1; sleep 2
  echo "[verify] attempt $try: serve $PRESET"
  : > "$SERVE"
  podman exec -d "$CN" bash -lc "export AITER_MOREH_ROOT_DIR='$AITER'; vllm-moreh serve '$MODEL' '$PRESET' > '$SERVE' 2>&1"
  # wait health or death (bounded)
  t=0; ok=0
  while [ "$t" -lt 900 ]; do
    if podman exec "$CN" bash -lc 'curl -sf http://localhost:8000/health >/dev/null 2>&1'; then ok=1; break; fi
    running || { echo "[verify] container down mid-serve"; break; }
    grep -qaE 'server died|Traceback|Error: |CUDA error|out of memory' "$SERVE" 2>/dev/null && { echo "[verify] serve error detected"; }
    sleep 15; t=$((t+15))
  done
  if [ "$ok" != 1 ]; then echo "[verify] not healthy (attempt $try) -> retry"; podman exec "$CN" bash -lc 'pkill -9 VLLM 2>/dev/null; true' >/dev/null 2>&1; sleep 20; continue; fi
  echo "[verify] HEALTH OK -> inspect EPLB engagement"
  sleep 10
  eplb_agents=$(grep -acE 'Initialized NIXL agent: eplb-' "$SERVE" 2>/dev/null)
  eplb_rank=$(grep -acE 'EPLB rank|enable_eplb|Rearranging experts|eplb_state' "$SERVE" 2>/dev/null)
  err_disable=$(grep -acE 'supports_eplb.*[Ff]alse|EPLB.*disabled|does not support' "$SERVE" 2>/dev/null)
  {
    echo "PATCH VERIFY VERDICT ($(date))"
    echo "container=$CN  image=moreh-vllm:0.23.0-260704-rc1  preset=$(basename "$PRESET")"
    echo "nixl eplb agents initialized : $eplb_agents (expect 8 -> EPLB engaged via supports_eplb)"
    echo "eplb engage markers          : $eplb_rank"
    echo "eplb-disabled markers        : $err_disable (expect 0)"
    if [ "${eplb_agents:-0}" -ge 1 ] && [ "${err_disable:-0}" -eq 0 ]; then
      echo "RESULT: PASS -> patch works: supports_eplb enabled EPLB (nixl communicator active)."
    else
      echo "RESULT: CHECK -> EPLB did not clearly engage; inspect $SERVE"
    fi
  } | tee "$VERD"
  podman exec "$CN" bash -lc 'pkill -9 VLLM 2>/dev/null; true' >/dev/null 2>&1
  exit 0
done
echo "[verify] gave up after $MAX attempts" | tee "$VERD"