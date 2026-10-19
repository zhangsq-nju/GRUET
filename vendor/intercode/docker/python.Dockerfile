FROM python:3.9-slim

WORKDIR /

RUN pip install rpyc

COPY docker/utils/python_server.py /

CMD ["python", "python_server.py"]
