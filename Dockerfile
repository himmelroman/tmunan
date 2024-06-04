FROM himmelroman/stream-diffusion:img2img

USER root
WORKDIR $HOME/app
RUN cd $HOME/app &&  \
    git clone https://github.com/himmelroman/tmunan.git

RUN $HOME/app && \
    pip install -r $HOME/app/tmunan/requirements_stream.txt
π
CMD ["uvicorn",  "tmunan.imagine.imagine_app:app", "--host", "0.0.0.0", "--port", "8080"]
