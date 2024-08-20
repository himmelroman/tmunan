FROM himmelroman/stream-diffusion:base

USER root
WORKDIR /root/app

RUN pip uninstall -y streamdiffusion && \
    pip install git+https://github.com/himmelroman/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]

# install requirements
COPY ./imagine_app/requirements.txt /root/app/requirements.txt
RUN pip install -r /root/app/requirements.txt

# set env
ENV HF_HOME=/app/models

# copy app
COPY . /root/app/tmunan
RUN chmod +x /root/app/run.sh

# command
ENTRYPOINT ["/root/app/run.sh"]
