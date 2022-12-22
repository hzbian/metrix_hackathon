FROM --platform=linux/amd64 ubuntu:20.04
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update && apt install -y gcc-7 g++-7 gfortran-7 libqwt-qt5-6 xvfb qt5-default python3-pip hdf5-tools
RUN pip install tqdm numpy pandas h5py

ENV DISPLAY :99
ADD run.sh /opt/run.sh
RUN chmod a+x /opt/run.sh

ADD RAY-UI-development /opt/ray-ui
RUN chmod a+x /opt/ray-ui/rayui
RUN mkdir -p /opt/ray-ui-workdir

# Install python3.10
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.10
RUN apt install -y python3-pip
RUN pip install virtualenv
RUN virtualenv --python="/usr/bin/python3.10" "/venv/"
ENV PATH="/venv/bin:$PATH"

ADD script_rayui_bg.py /opt/script_rayui_bg.py

RUN apt install -y mc

ENTRYPOINT ["tail", "-f", "/dev/null"]
