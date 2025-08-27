FROM python:3.13

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements requirements
RUN pip install -r requirements/ml.txt

# Install Jupyter kernel dependencies (for VS Code raw kernel support)
RUN pip install --upgrade pip && \
    pip install ipykernel jupyter_client pyzmq tornado traitlets debugpy nest-asyncio psutil && \
    python -m ipykernel install --user --name devcontainer-py --display-name "Python (Dev Container)"


RUN pip install -r requirements/extra.txt

WORKDIR /workspace