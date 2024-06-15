FROM himmelroman/stream-diffusion:mine

USER root
WORKDIR /root/app
RUN cd /root/app &&  \
    git clone https://github.com/himmelroman/tmunan.git

RUN cd /root/app && \
    pip install -r /root/app/tmunan/requirements_stream.txt

ENV HF_HOME=/root/app/models
WORKDIR /root/app/tmunan
CMD ["uvicorn",  "tmunan.imagine.stream_app:app", "--host", "0.0.0.0", "--port", "8080"]
