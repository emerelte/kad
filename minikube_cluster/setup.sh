#!/bin/bash

#echo "Starting docker deamon"
#
#rm /var/run/docker.pid 2>/dev/null
#dockerd &> /dev/null &
#docker ps &> /dev/null
#
#while [ $? -ne 0 ]
#do
#    echo "Waiting for docker deamon..."
#    sleep 1
#    docker ps &> /dev/null
#done
#
#echo "Docker deamon ready!"

# if set to 1 create admin account
#create_admin=1

#cluster_name=$(cat ./CLUSTERNAME)

minikube delete
minikube start

mkdir secret
cat ~/.kube/config > ./secret/kubeconfig

#sed -i -e 's|server:.*|server: http://127.0.0.1:8001|g' ./secret/kubeconfig

#if [ $create_admin == 1 ]; then
#    echo "Admin account created"
#
#    kubectl create -n kube-system serviceaccount admin
#    kubectl create clusterrolebinding permissive-binding \
#     --clusterrole=cluster-admin \
#     --user=admin \
#     --user=kubelet \
#     --group=system:serviceaccounts
#
#     admin_token_name=$(kubectl -n kube-system get serviceaccount admin -o yaml | tail -1 | cut -d":" -f 2 | cut -d" " -f2)
#     admin_token=$(kubectl -n kube-system get secret $admin_token_name -o yaml | grep token | head -1 | sed -e 's/.* \(.*\)$/\1/')
#     echo "Admin token: "
#     echo $admin_token | base64 -d
#     echo ""
#else
#    echo "Admin account not created"
#fi

echo "Starting sock-shop"
kubectl apply -f ../microservices-demo-master/deploy/kubernetes/complete-demo.yaml

echo "Starting monitoring"
kubectl apply -f ../microservices-demo-master/deploy/kubernetes/manifests-monitoring

echo "Waiting for pods to be ready"
kubectl wait --for=condition=ready pod -n sock-shop --all --timeout=120s
kubectl wait --for=condition=ready pod -n monitoring --all --timeout=120s

echo "Port forwarding"
kubectl port-forward service/prometheus 9090 -n monitoring &
kubectl port-forward service/front-end 30001:80 -n sock-shop &
kubectl port-forward service/grafana 31300:80 -n monitoring &

echo "Annotate sock-shop services to be scraped by Prometheus"
kubectl annotate service -n sock-shop prometheus.io/scrape='true' --all

# make local images available for minikube
eval $(minikube docker-env)

echo "------------------------"

minikube dashboard

exec $@
