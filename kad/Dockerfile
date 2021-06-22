FROM python:3.8-slim-buster

WORKDIR /kad

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /

CMD [ "python3", "kad/main.py" ]