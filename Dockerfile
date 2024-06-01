FROM nvcr.io/nvidia/pytorch:24.04-py3 as builder

# Build onnxruntime
RUN git clone --depth 1 --branch v1.17.3 https://github.com/microsoft/onnxruntime &&  \
    cd onnxruntime && \
    export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && \
    git config --global --add safe.directory '*' && \
    sh build.sh  \
            --config Release --build_shared_lib \
            --parallel 4 --nvcc_threads 1 \
            --use_cuda --cuda_version 12.4 \
            --cuda_home /usr/local/cuda-12.4 --cudnn_home /usr/lib/x86_64-linux-gnu/ \
            --use_tensorrt --tensorrt_home /usr/src/tensorrt \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=86 \
            --build_wheel --skip_tests \
            --allow_running_as_root

FROM nvcr.io/nvidia/pytorch:24.04-py3 as runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

#RUN apt-get update && apt-get install --no-install-recommends -y \
#    build-essential \
#    python3.9 \
#    python3-pip \
#    python3-dev \
#    git \
#    ffmpeg \
#    google-perftools \
#    ca-certificates curl gnupg \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Install onnxruntime
COPY --from=builder /workspace/onnxruntime/build/Linux/Release/dist/ /opt/dist/
RUN pip install --no-cache-dir /opt/dist/onnxruntime_gpu-*.whl --force-reinstall

# Install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Set the working directory to the user's home directory
WORKDIR $HOME/app
