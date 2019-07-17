FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update && apt-get install -y git

RUN pip install keras
# RUN apt-get install vim -y

# RUN pip install --upgrade pip

RUN pip install h5py scikit-optimize

RUN pip install matplotlib

# RUN git clone https://github.com/jobpasin/tooth-2d

# RUN export CUDA_VISIBLE_DEVICES=1

RUN printenv

WORKDIR pointNet

CMD ls

# CMD python3 train_cls.py