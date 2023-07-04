FROM --platform=linux/amd64 stateoftheartio/qt6:6.4-gcc-aqt
USER root
RUN sudo mkdir /usr/bin/gfortran
RUN apt update && apt -y install libglu1-mesa-dev freeglut3-dev mesa-common-dev wget git gfortran

RUN wget https://sourceforge.net/projects/qwt/files/qwt/6.2.0/qwt-6.2.0.tar.bz2/download
RUN tar -xf download
RUN cd qwt-6.2.0 && qmake qwt.pro && make -j $(nproc) && sudo make install
RUN sudo ln -s /usr/local/qwt-6.2.0/include/ /usr/include/qwt
RUN sudo ln -s /usr/local/qwt-6.2.0/lib/libqwt.so /usr/lib/libqwt.so
RUN sudo ln -s /usr/local/qwt-6.2.0/lib/libqwt.so /usr/lib/libqwt.so.6.2

RUN git clone https://docker:6sVD8PgJ_HCy8z7f74p2@git.ies.uni-kassel.de/dmeier/ray-ui.git RAY
RUN git config --global --add safe.directory /home/user/RAY
RUN cd RAY/src/Ray-UI && qmake Ray-UI.pro && make -j4
RUN sudo ln -s /home/user/RAY/build/build-Ray-UI-Linux---Qt_6_4_0-GCC_64bit-Release/Ray-UI /usr/bin/Ray-UI
RUN sudo ln -s /home/user/RAY/build/build-Ray-UI-Linux---Qt_6_4_0-GCC_64bit-Release/Ray-UI /usr/bin/ray-ui

ENV QT_QPA_PLATFORM=offscreen
RUN apt update && apt install -y python3
ADD script_rayui_bg.py /opt/script_rayui_bg.py
