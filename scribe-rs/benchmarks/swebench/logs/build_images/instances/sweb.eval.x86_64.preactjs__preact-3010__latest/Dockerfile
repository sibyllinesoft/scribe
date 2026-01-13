FROM --platform=linux/x86_64 sweb.env.js.x86_64.e48de95e424f48bce473c9:latest

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN node -v
RUN npm -v
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
