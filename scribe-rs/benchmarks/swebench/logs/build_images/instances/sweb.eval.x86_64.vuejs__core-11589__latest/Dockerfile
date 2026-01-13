FROM --platform=linux/x86_64 sweb.env.js.x86_64.6e41b0b889deef7c12bbd2:latest

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN node -v
RUN npm -v
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
