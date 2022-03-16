FROM pytorch/pytorch

# Install git
RUN apt-get update && apt-get install -y git

# Install dependencies
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache python -m pip install -r /requirements.txt

# Create non-root devcontainer user
RUN adduser --disabled-password --gecos "" devcontainer
RUN echo "devcontainer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER devcontainer