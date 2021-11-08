FROM python:3.8-slim-buster

# Install locust
RUN pip install pyzmq locust faker

RUN mkdir /config
COPY loads /config

RUN ls config/cyclic_with_jumpsup

WORKDIR /config/cyclic_with_jumpsup

CMD ["locust", "--host", "http://localhost:30001/", "--headless"]
