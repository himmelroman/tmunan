FROM himmelroman/stream-diffusion:mine

USER root
WORKDIR /app

RUN pip uninstall -y streamdiffusion && \
    pip install git+https://github.com/himmelroman/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]

COPY requirements_stream.txt /root/app/requirements_stream.txt

RUN cd /app && \
    pip install -r /root/app/requirements_stream.txt

RUN cd /app &&  \
    git clone https://github.com/himmelroman/tmunan.git

ENV HF_HOME=/app/models
WORKDIR /app/tmunan
CMD ["uvicorn",  "tmunan.imagine.stream_app:app", "--host", "0.0.0.0", "--port", "8080"]
