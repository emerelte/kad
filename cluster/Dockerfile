FROM docker:19.03.8-dind

# install tools
RUN apk add -U wget curl bash openssl jq

# install kubernetes
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl
RUN kubectl version --client

#install helm
RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
RUN chmod 700 get_helm.sh
RUN ./get_helm.sh

#install kind
RUN curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.8.1/kind-$(uname)-amd64
RUN chmod +x ./kind
RUN mv ./kind /usr/local/bin/kind
RUN kind --version

WORKDIR /home
COPY ./cluster-config.yaml .
COPY ./CLUSTERNAME .
COPY ./setup.sh .

ENTRYPOINT [ "./setup.sh" ]
CMD ["kubectl", "proxy", "--address=0.0.0.0"]
