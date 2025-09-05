#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:23.12-py3
# je tam zavilsost na 2.0.1
# islo by to na novsom?
RUN apt-get update && apt-get install -y --no-install-recommends  gdb git vim libaio-dev python3.10-venv
WORKDIR /src/
# generic requ.
COPY requirements.txt /src/
RUN pip install -r requirements.txt
# bitsandbytes
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
# pip install -r requirements-dev.txt && \
RUN cd bitsandbytes  && \ 
cmake -DCOMPUTE_BACKEND=cuda -S . && \
make && \
pip install -e .   
# for deepspeed
RUN pip install deepspeed mpi4py
# optimum
RUN pip install optimum
