FROM ubuntu/python:3.10-22.04_stable

USER ubuntu
WORKDIR /ubuntu/app

COPY ./stream_app/requirements.txt /ubuntu/app/requirements.txt

RUN pip install -r /ubuntu/app/requirements.txt

COPY . /ubuntu/app/tmunan

CMD ["uvicorn",  "tmunan.stream_app.stream:app", "--host", "0.0.0.0", "--port", "8080"]
