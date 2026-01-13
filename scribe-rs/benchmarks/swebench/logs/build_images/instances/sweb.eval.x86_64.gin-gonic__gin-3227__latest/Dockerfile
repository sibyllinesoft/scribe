FROM --platform=linux/x86_64 sweb.env.go.x86_64.afe3e7ce1226ce8865287b:latest

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
