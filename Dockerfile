FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update -y

WORKDIR /workspace/repository

COPY . /workspace/repository/

#fix for transformers import
RUN pip install -r requirements.txt && RUN pip uninstall transformer-engine -y

EXPOSE 8080 9090

CMD [ "python", "app/app.py"]