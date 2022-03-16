FROM pytorch/pytorch

# Install git
RUN apt-get update && apt-get install -y git

# Create non-root devcontainer user
RUN adduser --disabled-password --gecos "" devcontainer
RUN echo "devcontainer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER devcontainer