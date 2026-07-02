WORKING_DIR=/home/phuc-nguyen/workspaces/mv-4571
CONTAINER_NAME=phuc-nguyen-mv-4571
SETUP_DEV=${SETUP_DEV:-0}

if [ "$SETUP_DEV" = "1" ] && [ ! -d "$WORKING_DIR/vllm-moreh" ]; then
  git clone --recursive git@github.com:moreh-dev/vllm-moreh.git "$WORKING_DIR/vllm-moreh"
fi

VLLM_MOREH_DIR="$WORKING_DIR/vllm-moreh"

# Derive the dev image tag from the pinned vllm submodule version instead of
# hardcoding it. The base image ships a prebuilt vllm/torch + matching build
# deps; keeping its tag in sync with 3rdparty/vllm avoids image-vs-submodule
# mismatches. Mirrors the README workflow (.env is for `docker compose`;
# generated here for consistency but unused by this docker run).
DEV_IMAGE="vllm/vllm-openai-rocm:v0.23.0"
if [ -x "$VLLM_MOREH_DIR/scripts/utils/get_vllm_version.sh" ]; then
  VLLM_VERSION="$("$VLLM_MOREH_DIR/scripts/utils/get_vllm_version.sh")"
  if [ -n "$VLLM_VERSION" ]; then
    DEV_IMAGE="vllm/vllm-openai-rocm:v${VLLM_VERSION}"
    ( cd "$VLLM_MOREH_DIR" && VLLM_VERSION="$VLLM_VERSION" envsubst '$VLLM_VERSION' < .env.example > .env )
  fi
fi
DEV_IMAGE="255250787067.dkr.ecr.ap-northeast-2.amazonaws.com/unencrypted/moreh-vllm:0.23.0-260622-rc1"
echo "Using dev image: $DEV_IMAGE"

docker run -d \
  --ipc=host --network=host --group-add render \
  --privileged --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v /remote/vast0/share-mv:/remote/vast0/share-mv \
  -v /home/phuc-nguyen/.claude:/root/.claude \
  -v $WORKING_DIR:$WORKING_DIR \
  -w $WORKING_DIR \
  -e AITER_MOREH_ROOT_DIR=$WORKING_DIR/vllm-moreh/src/aiter_moreh \
  --name $CONTAINER_NAME \
  --entrypoint bash \
  "$DEV_IMAGE" \
  -lc 'sleep infinity'

if [ "$SETUP_DEV" = "1" ]; then
  docker exec -ti $CONTAINER_NAME bash -lc \
    "git config --global --add safe.directory '*' && cd $WORKING_DIR/vllm-moreh && WORKING_DIR=$WORKING_DIR/vllm-moreh source scripts/utils/setup_dev.sh; exec bash"
else
  docker exec -ti $CONTAINER_NAME bash
fi
