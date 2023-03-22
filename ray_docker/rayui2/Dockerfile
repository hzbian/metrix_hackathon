FROM --platform=linux/amd64 stateoftheartio/qt6:6.4-gcc-aqt
RUN sudo apt update

RUN sudo mkdir /usr/bin/gfortran
RUN sudo apt -y install libglu1-mesa-dev freeglut3-dev mesa-common-dev wget git gfortran

RUN wget https://sourceforge.net/projects/qwt/files/qwt/6.2.0/qwt-6.2.0.tar.bz2/download
RUN tar -xf download
RUN cd qwt-6.2.0 && qmake qwt.pro && make -j4 && sudo make install
RUN sudo ln -s /usr/local/qwt-6.2.0/include/ /usr/include/qwt
RUN sudo ln -s /usr/local/qwt-6.2.0/lib/libqwt.so /usr/lib/libqwt.so
RUN sudo ln -s /usr/local/qwt-6.2.0/lib/libqwt.so /usr/lib/libqwt.so.6.2

RUN git clone https://docker:glpat-KpVX4NHmimF2nseAKJRo@gitlab.helmholtz-berlin.de/RAY/RAY.git
RUN git config --global --add safe.directory /home/user/RAY
RUN cd RAY/src/Ray-UI && qmake Ray-UI.pro && make -j4

# Install python3.10
RUN sudo apt update
RUN sudo apt install -y software-properties-common
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN sudo apt install -y python3.10
RUN sudo apt install -y python3-pip
RUN sudo pip install virtualenv
RUN sudo virtualenv --python="/usr/bin/python3.10" "/venv/"
ENV PATH="/venv/bin:$PATH"

ADD script_rayui_bg.py /opt/script_rayui_bg.py

ENTRYPOINT ["QT_QPA_PLATFORM=offscreen", "/home/user/RAY/build/build-Ray-UI-Linux---Qt_6_4_0-GCC_64bit-Release/Ray-UI", "-b"]
