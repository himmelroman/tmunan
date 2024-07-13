git clone https://github.com/himmelroman/tmunan.git
cd tmunan
pip install -r requirements.txt

screen -R tmunan
mkdir -p /home/ubuntu/.cache/theatre/imagine/rubin_style
curl https://b5ff-62-56-134-6.ngrok-free.app/rubin3.jpg > /home/ubuntu/.cache/theatre/imagine/rubin_style/rubin3.jpg
CUDA_MODEL_SIZE=small uvicorn tmunan.imagine.imagine_app:app --host 0.0.0.0 --port 8080

# Run on GPU server with CUDA
docker run -ti --rm -v ~/.cache/huggingface:/home/user/.cache/huggingface -p 8080:8080 --gpus all himmelroman/stream-diffusion:stream


#  1  vi sd.py
#  2  python sd.py
#  3  python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
#  4  python -m pip install --upgrade-strategy eager optimum[neuronx]
#  5  python sd.py
#  6  pip install diffusers
#  7  python sd.py
#  8  history