FROM python:3.9

# FROM python:3.9-alpine


ENV SHADOW_APP_NAME='app.hello_world'

WORKDIR /shadow

COPY pyproject.toml ./
COPY src/ ./src

# install main shadow scholar package, as weel as dependencies for this
# specific app
RUN pip install .
RUN pip install "$(shadow -r ${SHADOW_APP_NAME})"

# Kick things off
CMD [ "sh", "-c", "shadow -c app/config.json $SHADOW_APP_NAME" ]
