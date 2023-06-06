#!/bin/bash
set -e
set -u

USERNAME=$(whoami)
docker run -it \
	--mount type=bind,source="$(pwd)",target=/home/${USERNAME}/sketch_transformer \
	--mount type=bind,source="/data1/honda/results",target=/home/${USERNAME}/sketch_transformer/results \
	--user=$(id -u $USER):$(id -g $USER) \
	--env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--rm \
	--gpus=all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--workdir=/home/${USERNAME}/sketch_transformer \
	--name=${USERNAME}_isaacgym_container_ ${USERNAME}/isaacgym /bin/bash