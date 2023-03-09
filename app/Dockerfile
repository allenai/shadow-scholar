FROM python:3.9

# name of the shadow scholar app to run
ENV SHADOW_APP_NAME='app.hello_world'

# directory where the code is from docker-compose
WORKDIR /shadow

# copy over the shadow scholar app code so we can pip install it
COPY pyproject.toml ./
COPY src/ ./src
COPY app/config.json ./app/config.json

# install main shadow scholar package, as weel as dependencies
# for this specific app (which I get with -r followed by the app name)
RUN pip install .
RUN pip install "$(shadow -r ${SHADOW_APP_NAME})"

# Kick things off
CMD [ "sh", "-c", "shadow -c app/config.json $SHADOW_APP_NAME" ]
