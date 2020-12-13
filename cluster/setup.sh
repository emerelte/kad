#!/bin/bash

echo "Starting docker deamon"

rm /var/run/docker.pid 2>/dev/null
dockerd &> /dev/null &
docker ps &> /dev/null

while [ $? -ne 0 ]
do
    echo "Waiting for docker deamon..."
    sleep 1
    docker ps &> /dev/null
done

echo "Docker deamon ready!"

# script creates a Kubernetes cluser from scratch
# configuration file is cluser-config.yaml

# if set to 1 create admin account
create_admin=1

cluster_name="kind"
cluster_name=$(cat ./CLUSTERNAME)

kind delete cluster --name $cluster_name
kind create cluster --config cluster-config.yaml --name $cluster_name

mkdir secret
kind get kubeconfig --name $cluster_name > ./secret/kubeconfig

sed -i -e 's|server:.*|server: http://127.0.0.1:8001|g' ./secret/kubeconfig

if [ $create_admin == 1 ]; then
    echo "Admin account created"
    
    kubectl create -n kube-system serviceaccount admin
    kubectl create clusterrolebinding permissive-binding \
     --clusterrole=cluster-admin \
     --user=admin \
     --user=kubelet \
     --group=system:serviceaccounts

     admin_token_name=$(kubectl -n kube-system get serviceaccount admin -o yaml | tail -1 | cut -d":" -f 2 | cut -d" " -f2)
     admin_token=$(kubectl -n kube-system get secret $admin_token_name -o yaml | grep token | head -1 | sed -e 's/.* \(.*\)$/\1/')
     echo "Admin token: "
     echo $admin_token | base64 -d
     echo ""
else
    echo "Admin account not created"
fi

echo "Installing Prometheus"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add stable https://kubernetes-charts.storage.googleapis.com/
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack --version "9.4.1"

echo "------------------------"

exec $@
