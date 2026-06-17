WORKING_DIR=/home/phuc-nguyen/workspace/mv-4476
CONTAINER_NAME=phuc-nguyen-mv-4476
SETUP_DEV=${SETUP_DEV:-0}

if [ "$SETUP_DEV" = "1" ] && [ ! -d "$WORKING_DIR/vllm-moreh" ]; then
  git clone --recursive https://github.com/moreh-dev/vllm-moreh.git "$WORKING_DIR/vllm-moreh"
fi

docker run -d \
  --ipc=host --network=host --group-add render \
  --privileged --security-opt seccomp=unconfined \
  --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  -v /remote/vast0/share-mv:/remote/vast0/share-mv \
  -v $WORKING_DIR:$WORKING_DIR \
  -w $WORKING_DIR \
  -e AITER_MOREH_ROOT_DIR=$WORKING_DIR/vllm-moreh/src/aiter_moreh \
  --name $CONTAINER_NAME \
  --entrypoint bash \
  vllm/vllm-openai-rocm:v0.21.0 \
  -lc 'sleep infinity'

if [ "$SETUP_DEV" = "1" ]; then
  docker exec -ti $CONTAINER_NAME bash -lc \
    "git config --global --add safe.directory '*' && cd $WORKING_DIR/vllm-moreh && WORKING_DIR=$WORKING_DIR/vllm-moreh source scripts/utils/setup_dev.sh; exec bash"
else
  docker exec -ti $CONTAINER_NAME bash
fi
