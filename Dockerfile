FROM himmelroman/stream-diffusion:img2img

RUN mkdir $HOME/app && cd $HOME/app
RUN git clone https://github.com/himmelroman/tmunan.git

RUN pip install -r tmunan/requirements_stream.txt

CMD ["uvicorn",  "tmunan.imagine.imagine_app:app", "--host", "0.0.0.0", "--port", "8080"]
