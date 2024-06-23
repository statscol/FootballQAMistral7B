FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update -y && pip install "huggingface_hub[cli]"

WORKDIR /workspace/repository

COPY . /workspace/repository/

RUN pip install -r requirements.txt

EXPOSE 8080 9090

CMD [ "python", "app/app.py"]