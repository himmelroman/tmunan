git clone https://github.com/himmelroman/tmunan.git
cd tmunan
pip install -r requirements.txt

screen -R tmunan
CUDA_MODEL_SIZE=small uvicorn tmunan.imagine.imagine_app:app --host 0.0.0.0 --port 8080
