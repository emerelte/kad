#!/bin/bash

# shellcheck disable=SC2046
eval $(minikube docker-env)

echo "Building KAD docker image"
docker build -t kad .

echo "Applying configmap"
kubectl apply -f kad-configmap.yaml

echo "Starting KAD"
kubectl apply -f kad-deployment.yaml

echo "Applying service"
kubectl apply -f kad-service.yaml

echo "Waiting for pods to be ready"
kubectl wait --for=condition=ready pod -n kad --all --timeout=120s

echo "Port forwarding"
kubectl port-forward service/kad 5000:5000 -n kad &
