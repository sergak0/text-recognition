FROM anibali/pytorch:1.10.0-cuda11.3-ubuntu20.04

USER root
ENV TZ=Asia/Tokyo
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update
RUN sudo apt-get install ffmpeg libsm6 libxext6 gcc -y

USER user

RUN pip install albumentations==0.5.2 numpy==1.21.4 opencv_python==4.5.4.60 segmentation_models_pytorch==0.2.1 transformers sentencepiece pandas torchvision numba

RUN pip install --upgrade pip && pip install wheel

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html

RUN git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection && pip install -e . && python setup.py develop


WORKDIR /workspace
