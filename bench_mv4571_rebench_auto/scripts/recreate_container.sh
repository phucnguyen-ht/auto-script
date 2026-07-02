#!/usr/bin/env bash
# Recreate the rebench container with vllm + vllm_moreh SOURCE mounted editable
# (copied out of the current image once), so we can instrument/patch the EPLB
# async deadlock and re-serve without rebuilding. Run ON THE GPU NODE (podman).
#   ssh gpu-5 'bash <this>'            # copy pkgs (once) + (re)create dev container
#   FORCE_COPY=1 ...                   # re-copy pkgs even if present
# Then edit src under $SRC_DIR/{vllm,vllm_moreh}/**.py and just re-serve.
set -euo pipefail

IMAGE="${IMAGE:-255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.23.0-260626-rc1}"
SRC_CONTAINER="${SRC_CONTAINER:-phuc-nguyen-mv4571-rebench}"   # copy packages from here
NEW_CONTAINER="${NEW_CONTAINER:-phuc-nguyen-mv4571-rebench-dev}"
WORKING_DIR="${WORKING_DIR:-/shared/amdgpu/home/loc_tran_ce6/phucnguyen/mv-4571}"
SRC_DIR="${SRC_DIR:-${WORKING_DIR}/auto-script/bench_mv4571_rebench_auto/vllm_src}"
DP=/usr/local/lib/python3.12/dist-packages
PKGS=(vllm vllm_moreh)

# 1. Copy packages out of the source container (once; editable copy on the host).
mkdir -p "${SRC_DIR}"
for p in "${PKGS[@]}"; do
    if [ "${FORCE_COPY:-0}" = 1 ] || [ ! -d "${SRC_DIR}/${p}" ]; then
        echo "[recreate] copy ${p} <- ${SRC_CONTAINER}:${DP}/${p}"
        rm -rf "${SRC_DIR}/${p}"
        podman cp "${SRC_CONTAINER}:${DP}/${p}" "${SRC_DIR}/${p}"
    else
        echo "[recreate] keep existing ${SRC_DIR}/${p} (FORCE_COPY=1 to refresh)"
    fi
done

# 2. (Re)create the dev container, mounting the editable packages over dist-packages.
podman rm -f "${NEW_CONTAINER}" 2>/dev/null || true
MOUNTS=(); for p in "${PKGS[@]}"; do MOUNTS+=(-v "${SRC_DIR}/${p}:${DP}/${p}"); done
podman run -ti -d \
    --ipc=host --network=host --group-add render --privileged \
    --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -v "/shared/amdgpu/home/loc_tran_ce6/share-mv:/remote/vast0/share-mv" \
    -v "${WORKING_DIR}:${WORKING_DIR}" \
    "${MOUNTS[@]}" \
    -w "${WORKING_DIR}" \
    --name "${NEW_CONTAINER}" \
    -e PYTORCH_ROCM_ARCH="gfx942" \
    --entrypoint /bin/bash \
    "${IMAGE}"

echo "[recreate] ${NEW_CONTAINER} up; editable src -> ${SRC_DIR}/{vllm,vllm_moreh}"
echo "[recreate] edit *.py there, then re-serve in ${NEW_CONTAINER} (pure-python edits take effect immediately; .so needs rebuild)."
