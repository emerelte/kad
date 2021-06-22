FROM docker:19.03.8-dind

# install tools
RUN apk add -U wget curl bash openssl jq

# install kubernetes
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl
RUN kubectl version --client

# install minikube
RUN curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
RUN sudo install minikube-linux-amd64 /usr/local/bin/minikube

WORKDIR /home
COPY ./setup.sh .

ENTRYPOINT [ "./setup.sh" ]
CMD ["kubectl", "proxy", "--address=0.0.0.0"]
