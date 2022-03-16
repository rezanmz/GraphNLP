FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04 

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

# Install python dependencies
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r /requirements.txt

# Create non-root devcontainer user
RUN adduser --disabled-password --gecos "" devcontainer
RUN echo "devcontainer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER devcontainer