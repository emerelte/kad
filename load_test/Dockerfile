FROM python:3.8-slim-buster

# Install locust
RUN pip install pyzmq locust faker

RUN mkdir /config
COPY . /config

WORKDIR /config

CMD ["locust", "--host", "http://localhost:30001/", "--headless"]
