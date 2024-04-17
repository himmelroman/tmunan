git clone https://github.com/himmelroman/tmunan.git
cd tmunan
pip install -r requirements.txt

screen -R tmunan
mkdir -p /home/ubuntu/.cache/theatre/imagine/rubin_style
curl https://b5ff-62-56-134-6.ngrok-free.app/rubin3.jpg > /home/ubuntu/.cache/theatre/imagine/rubin_style/rubin3.jpg
CUDA_MODEL_SIZE=small uvicorn tmunan.imagine.imagine_app:app --host 0.0.0.0 --port 8080
