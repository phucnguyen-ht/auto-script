IMAGE="255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.17.0-260422-dccf357-hsa"
WORKING_DIR="/shared/amdgpu/home/loc_tran_ce6/phucnguyen"
CONTAINER_NAME="phuc-nguyen"
SETUP_DEV=${SETUP_DEV:-0}

if [ "$SETUP_DEV" = "1" ] && [ ! -d "$WORKING_DIR/vllm-moreh" ]; then
  git clone --recursive git@github.com:moreh-dev/vllm-moreh.git "$WORKING_DIR/vllm-moreh"
fi

podman run -ti -d \
  --ipc=host --network=host \
  --group-add render \
  --privileged \
  --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v "/shared/amdgpu/home/loc_tran_ce6/share-mv:/remote/vast0/share-mv" \
  -v $WORKING_DIR:$WORKING_DIR \
  -w $WORKING_DIR \
  --name "${CONTAINER_NAME}" \
  -e PYTORCH_ROCM_ARCH="gfx942" \
  --entrypoint /bin/bash \
  "${IMAGE}"

if [ "$SETUP_DEV" = "1" ]; then
  podman exec -ti "${CONTAINER_NAME}" bash -lc \
    "git config --global --add safe.directory '*' && cd $WORKING_DIR/vllm-moreh && WORKING_DIR=$WORKING_DIR/vllm-moreh source scripts/utils/setup_dev.sh; exec bash"
else
  podman exec -ti "${CONTAINER_NAME}" bash
fi
