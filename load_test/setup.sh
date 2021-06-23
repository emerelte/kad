#!/bin/bash

# shellcheck disable=SC2046
eval $(minikube docker-env)

echo "Building load-test docker image"
docker build -t load-test .

echo "Starting load test"
kubectl apply -f non-anomalous.yaml
