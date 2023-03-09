FROM python:3.9

# parsing some arguments as input; optional, but useful for local
# development before creating a separate marina branch
ARG shadow_app_name='app.hello_world'
ARG shadow_config_path='app/config.json'

# name of the shadow scholar app to run
ENV SHADOW_APP_NAME=${shadow_app_name}

# directory where the code is from docker-compose
WORKDIR /shadow

# copy over the shadow scholar app code so we can pip install it
COPY pyproject.toml ./
COPY MANIFEST.in ./
COPY src/ ./src
COPY ${shadow_config_path} ./app/config.json

# install main shadow scholar package, as weel as dependencies
# for this specific app (which I get with -r followed by the app name)
RUN pip install .
RUN pip install "$(shadow -r ${SHADOW_APP_NAME})"

# Kick things off
CMD [ "sh", "-c", "shadow -c app/config.json ${SHADOW_APP_NAME}" ]
