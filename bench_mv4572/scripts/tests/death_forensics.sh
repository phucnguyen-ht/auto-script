#!/usr/bin/env bash
# death_forensics.sh OUT — runs on the GPU host. Watches the vllm serve inside the PR
# container; every 5s logs serve-liveness + foreign GPU containers (police.sh) + GPU VRAM.
# On serve death, immediately dumps: foreign-container list, dmesg tail (OOM/signal), GPU
# mem — so we LOG the actual kill cause instead of guessing (per the evidence rule).
set -uo pipefail
OUT="${1:?usage: death_forensics.sh OUTFILE}"
CONT=phuc-nguyen-mv4572-pr-0.24.0
POLICE=/shared/amdgpu/home/loc_tran_ce6/share-mv/police.sh
exec >>"$OUT" 2>&1
echo "==================== forensics start $(date -u) (UTC) ===================="
seen_alive=0
while :; do
  ts=$(date -u '+%H:%M:%S')
  alive=$(podman exec "$CONT" bash -lc 'pgrep -cf "vllm serve" 2>/dev/null || echo 0' 2>/dev/null)
  alive=${alive:-0}
  foreign=$(bash "$POLICE" 2>/dev/null | grep -oE 'id=[0-9a-f]+' | wc -l)
  vram=$(rocm-smi --showmemuse 2>/dev/null | grep -cE 'GPU Memory Allocated \(VRAM%\)' )
  echo "$ts alive=$alive foreign_gpu_containers=$foreign vram_lines=$vram"
  if [ "$seen_alive" = "1" ] && [ "$alive" = "0" ]; then
    echo "!!!!!!!!!! SERVE DIED at $ts UTC — FORENSICS !!!!!!!!!!"
    echo "--- police.sh (foreign GPU users at death) ---"; bash "$POLICE" 2>/dev/null | head -20
    echo "--- rocm-smi mem ---"; rocm-smi --showmemuse 2>/dev/null | head -20
    echo "--- dmesg tail (host, via privileged container) ---"
    podman exec "$CONT" bash -lc 'dmesg -T 2>/dev/null | tail -30' 2>/dev/null || echo "(dmesg unavailable)"
    echo "--- host dmesg tail ---"; dmesg -T 2>/dev/null | tail -30 || echo "(host dmesg unavailable)"
    break
  fi
  [ "$alive" -ge 1 ] && seen_alive=1
  sleep 5
done
echo "==================== forensics end $(date -u) ===================="
