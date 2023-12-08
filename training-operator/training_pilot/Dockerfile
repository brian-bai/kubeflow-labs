FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt update && apt install -y libibverbs-dev
RUN apt install -y net-tools

RUN pip install protobuf==3.20.0
RUN pip install tensorboardX==1.6.0
RUN mkdir -p /opt/imgnet

WORKDIR /opt/imgnet/src
ADD imagenet.py /opt/imgnet/src/imgnet.py

RUN  chgrp -R 0 /opt/imgnet \
  && chmod -R g+rwX /opt/imgnet

ENTRYPOINT ["python", "/opt/imgnet/src/imgnet.py"]