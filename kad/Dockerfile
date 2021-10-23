FROM python:3.6.15-slim-buster

WORKDIR /kad

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /

ENV PYTHONPATH "${PYTHONPATH}:/"

CMD [ "python3", "kad/main.py" ]