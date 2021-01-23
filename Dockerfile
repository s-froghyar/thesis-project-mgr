FROM python:3.7
ARG USER
ARG API_KEY
RUN echo "machine api.packagr.app \
         "    login ${USER} \
         "    password ${API_KEY}" > /root/.netrc
RUN chown root ~/.netrc
RUN chmod 0600 ~/.netrc
COPY requirement.txt /requirements.txt
RUN pip install -r requirements.txt
COPY ./src .

WORKDIR /experiments/baseline
