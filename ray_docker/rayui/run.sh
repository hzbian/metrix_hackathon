#!/bin/bash
FILE=/tmp/.X99-lock
if [[ ! -f "$FILE" ]]; then
    Xvfb :99 -screen 0 640x480x8 -nolisten tcp &
fi
export LD_LIBRARY_PATH=/opt/ray-ui
/opt/ray-ui/rayui -b

