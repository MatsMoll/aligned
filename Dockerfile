FROM python:3.10.5-slim

WORKDIR /opt/app

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN pip install pip --upgrade

COPY ./pyproject.toml /opt/app/pyproject.toml
COPY ./poetry.lock /opt/app/poetry.lock

RUN mkdir /opt/app/aligned
RUN poetry install --no-dev --no-root --extras "redis psql server aws"

COPY ./aligned /opt/app/aligned

ENTRYPOINT ["python", "-m", "aligned.cli"]
