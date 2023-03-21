#!/bin/bash
FILE=/tmp/.X99-lock
if [[ ! -f "$FILE" ]]; then
    Xvfb :99 -screen 0 640x480x8 -nolisten tcp &
fi
QT_QPA_PLATFORM=offscreen /home/user/RAY/build/build-Ray-UI-Linux---Qt_6_4_0-GCC_64bit-Release/Ray-UI -b
