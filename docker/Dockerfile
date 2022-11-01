# Ubuntu Linux as the base image
FROM ubuntu:18.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pip python3-dev


# Install transformers

RUN pip3 install --upgrade pip
RUN pip3 install transformers
RUN pip3 install sentencepiece
RUN pip3 install torch torchvision

# Add the files into container, under QG folder
RUN mkdir QG
ADD ask.py /QG

# Change the permission of the programs
CMD ["chmod 777 /QG/*"]

# Set the working directory as QG

WORKDIR /QG
ENTRYPOINT ["/bin/bash", "-c" ]

