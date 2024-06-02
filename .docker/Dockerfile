FROM himmelroman/stream-diffusion:latest
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR $HOME/app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]
RUN python -m streamdiffusion.tools.install-tensorrt


USER user
CMD ["python",  "main.py", "--debug", "--acceleration", "tensorrt"]
