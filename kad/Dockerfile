FROM python:3.6.15-slim-buster

WORKDIR /kad

RUN apt-get update && apt-get install -y iputils-ping
RUN apt-get install -y vim
RUN apt-get install -y curl
RUN apt-get install -y procps

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /

ENV PYTHONPATH "${PYTHONPATH}:/"

CMD [ "python3", "kad/main.py" ]
