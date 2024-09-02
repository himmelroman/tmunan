FROM python:3.11

USER root
WORKDIR /root/app

COPY ./stream_app/requirements.txt /root/app/requirements.txt

RUN pip install -r /root/app/requirements.txt

COPY . /root/app/tmunan

CMD ["uvicorn",  "tmunan.stream_app.stream:app", "--host", "0.0.0.0", "--port", "8080"]
