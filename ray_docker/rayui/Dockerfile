FROM --platform=linux/amd64 ubuntu:20.04
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update && apt install -y gcc-7 g++-7 gfortran-7 libqwt-qt5-6 xvfb qt5-default python3-pip hdf5-tools
RUN pip install tqdm numpy pandas h5py

ADD RAY-UI-development /opt/ray-ui

ENV DISPLAY :99
ADD run.sh /run.sh
RUN chmod a+x /run.sh

#CMD cd /opt/traindata && python3 TrainingData_METRIX_V4.py
