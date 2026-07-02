#!/usr/bin/env bash
# Recreate the gpu-5 podman container `phuc-nguyen-mv-4571` with the patched 3rdparty vllm/vllm_moreh
# bind-mounted from auto-script (NOT the old auto-script-gpu-5). Run ON gpu-5 (or via ssh gpu-5).
# Faithful to the original inspect (image, privileged, host net/ipc, render group, /dev/{kfd,dri,mem}).
set -e
MV=/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571
AS=$MV/auto-script/bench_mv4571/auto_analyze_ep_imbalance/3rdparty
IMG=255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.23.0-260622-rc1

ls -d "$AS/vllm" "$AS/vllm_moreh" /shared/amdgpu/home/loc_tran_ce6/share-mv >/dev/null
podman rm -f phuc-nguyen-mv-4571 2>/dev/null || true
podman run -d \
  --ipc=host --network=host --group-add render \
  --privileged --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v /shared/amdgpu/home/loc_tran_ce6/share-mv:/remote/vast0/share-mv \
  -v "$MV:$MV" \
  -v "$AS/vllm:/usr/local/lib/python3.12/dist-packages/vllm" \
  -v "$AS/vllm_moreh:/usr/local/lib/python3.12/dist-packages/vllm_moreh" \
  -w "$MV" \
  --name phuc-nguyen-mv-4571 \
  --entrypoint bash \
  "$IMG" \
  -lc 'sleep infinity'
podman ps --format '{{.Names}} {{.Status}}' | grep phuc

# yq: harness needs mikefarah yq v4. The repo's own ensure_yq (common/helper.sh, pinned
# v4.50.1 + sha256 verify) installs it; the harness calls it automatically on first run,
# but install up-front so ad-hoc yq calls work too.
podman exec phuc-nguyen-mv-4571 bash -lc \
  'cd /shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571/auto-script && source common/helper.sh >/dev/null 2>&1; yq --version'
