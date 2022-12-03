FROM ubuntu:20.04

# Build warning for installing pip: docker build --network=host -t image_name .

# Keeps Image Small by not installing suggested or recommended dependencies
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y nano \
  && apt-get install -y python3.8 \
  && apt-get install -y python3-pip \
  && apt-get install -y git \
  && apt-get install -y openssh-client \
  && rm -rf /var/lib/apt/lists/*

# Creates a user other than root and logins
#RUN useradd -ms /bin/bash ml_machine
#USER ml_machine

WORKDIR /home
COPY requirements_py3.8.txt /home
RUN pip install -r requirements_py3.8.txt



