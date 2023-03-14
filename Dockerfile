FROM python:3.10.5-slim

WORKDIR /opt/app

COPY /. opt/app/aligned

RUN pip install -e 'opt/app/aligned[redis,aws,psql,server]'
