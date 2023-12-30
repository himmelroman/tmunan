FROM pytorch/pytorch:latest

WORKDIR /app/

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY ./tmunan /app/tmunan

CMD ["uvicorn", "tmunan.api.app:app", "--host", "0.0.0.0", "--port", "80"]
