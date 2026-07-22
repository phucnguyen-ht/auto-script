#!/usr/bin/env bash
# [MV-4572 Task1.1] Host launcher — chạy FRESH full-grid confirm sweep (4 config x 12 scenario)
# theo yêu cầu user "hãy chạy full case". CHAIN sau khi p1p2r0 (Task2) xong để KHÔNG tranh GPU.
# tmux window confirm024 trên mi355-gpu-58. Tee console.
set -uo pipefail
WD=/shared/amdgpu/home/loc_tran_ce6/phuc-nguyen/mv-4572
CT=phuc-nguyen-mv4572-analysis-0.24.0
LOGDIR="$WD/auto-script/bench_mv4572/logs_0.24.0_confirm"
mkdir -p "$LOGDIR"

# 1) Chờ Task2 p1p2r0 kết thúc (tránh race GPU). Bỏ qua nếu console không tồn tại.
P1P2="$WD/auto-script/bench_mv4572/logs_0.24.0_norearr/console_20260718_p1p2r0.log"
if [ -f "$P1P2" ]; then
  echo "[chain] chờ p1p2r0 (Task2) DONE trước khi chạy Task1.1 sweep... @$(date)"
  while ! grep -q "\[launch\] DONE" "$P1P2" 2>/dev/null; do sleep 30; done
  echo "[chain] p1p2r0 xong. Chờ thêm 60s cho GPU nhả sạch." ; sleep 60
fi

# 2) FRESH full-grid confirm sweep (Task 1.1, non-mtp). sweep_presets.sh tự wait_gpu_free trước mỗi serve.
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
CONSOLE="$LOGDIR/console_confirm_$TS.log"
echo "[chain] LAUNCH Task1.1 fresh full sweep (non-mtp) TS=$TS @$(date)" | tee "$CONSOLE"
podman exec -i "$CT" bash -lc "cd $WD && TS=$TS bash auto-script/bench_mv4572/scripts/confirm_v0.24.0.sh" 2>&1 | tee -a "$CONSOLE"
echo "[chain] Task1.1 confirm sweep DONE TS=$TS @$(date)" | tee -a "$CONSOLE"

# 3) FRESH full-grid MTP sweep (Task 1.2) — 4 config (base/r0 x V2 on/off). V2-ON MTP presets MỚI tạo.
TS_MTP="${TS}_mtp"
CONSOLE_MTP="$WD/auto-script/bench_mv4572/logs_0.24.0_confirm_mtp/console_confirm_$TS_MTP.log"
mkdir -p "$WD/auto-script/bench_mv4572/logs_0.24.0_confirm_mtp"
echo "[chain] LAUNCH Task1.2 fresh MTP full sweep (4 config) TS=$TS_MTP @$(date)" | tee "$CONSOLE_MTP"
podman exec -i "$CT" bash -lc "cd $WD && TS=$TS_MTP PRESET_LIST='base_v2.yaml r0_v2.yaml base_noV2.yaml r0_noV2.yaml' bash auto-script/bench_mv4572/scripts/confirm_v0.24.0_mtp.sh" 2>&1 | tee -a "$CONSOLE_MTP"
echo "[chain] Task1.2 MTP confirm sweep DONE TS=$TS_MTP @$(date)" | tee -a "$CONSOLE_MTP"
echo "[chain] ALL Task1 fresh sweeps DONE @$(date)"
