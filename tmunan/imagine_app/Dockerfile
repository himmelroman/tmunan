FROM himmelroman/stream-diffusion:mine

USER root
WORKDIR /root/app

RUN pip uninstall -y streamdiffusion && \
    pip install git+https://github.com/himmelroman/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]

COPY ./requirements.txt /root/app/requirements.txt

RUN pip install -r /root/app/requirements.txt

COPY . /root/app/

ENV HF_HOME=/app/models
CMD ["uvicorn",  "tmunan.imagine_app.main:app", "--host", "0.0.0.0", "--port", "8080"]
