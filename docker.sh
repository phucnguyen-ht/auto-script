IMAGE="255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.17.0-260422-dccf357-hsa"
WORKING_DIR="/shared/amdgpu/home/loc_tran_ce6/phucnguyen"
CONTAINER_NAME=$1

podman run -ti -d \
  --ipc=host --network=host \
  --group-add render \
  --privileged \
  --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v "/shared/amdgpu/home/loc_tran_ce6/share-mv/zai-org:/share-mv/zai-org" \
  -v $WORKING_DIR:$WORKING_DIR \
  -w $WORKING_DIR \
  --name "${CONTAINER_NAME}" \
  -e PYTORCH_ROCM_ARCH="gfx942" \
  --entrypoint /bin/bash \
  "${IMAGE}"

podman exec -ti "${CONTAINER_NAME}" bash
