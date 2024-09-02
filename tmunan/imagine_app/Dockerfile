FROM 090268630745.dkr.ecr.eu-west-1.amazonaws.com/tmunan/stream_diffusion:base

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

# command
ENTRYPOINT ["/root/app/tmunan/imagine_app/run.sh"]
