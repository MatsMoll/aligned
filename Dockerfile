FROM python:3.10.5-slim

WORKDIR /opt/app

COPY ./pyproject.toml /opt/app/pyproject.toml
# COPY ./poetry.lock /opt/app/poetry.lock

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN mkdir /opt/app/aligned
RUN poetry install --no-dev --no-root --extras "redis psql server aws"

COPY ./aligned /opt/app/aligned

# COPY /. opt/app/aligned

ENTRYPOINT ["python", "-m", "aligned.cli"]
# RUN pip install -U 'opt/app/aligned[redis,aws,psql,server,text]'
