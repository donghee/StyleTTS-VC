#!/bin/sh

xhost +local:root

# no gpu
docker run -it \
    --privileged \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}/..:/StyleTTS-VC" \
    -p 8899:8888 \
    --rm \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --name styletts-vc \
    styletts-vc:v1 \
    #segdinet_add_opendace_cu113_ubuntu20.04 \
    #jupyter notebook --ip 0.0.0.0 --allow-root
    bash
