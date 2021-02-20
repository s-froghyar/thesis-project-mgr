FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
RUN apt-get -y update
RUN apt-get install -y libsndfile1
WORKDIR /app
COPY requirements.txt .
RUN conda install -c conda-forge sox
RUN pip install -r requirements.txt

COPY runner.py .
COPY config ./config/
COPY model_fitter ./model_fitter/
COPY tpreporter ./tpreporter/


# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

# # installs python 3.7
# # https://stackoverflow.com/a/58562728

# # Upgrade installed packages
# RUN apt-get update && apt-get upgrade -y && apt-get clean

# # Python package management and basic dependencies
# RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils libsndfile1

# # Register the version in alternatives
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# # Set python 3 as the default python
# RUN update-alternatives --set python /usr/bin/python3.7

# # Upgrade pip to latest version
# RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python get-pip.py --force-reinstall && \
#     rm get-pip.py

# # installs python 3.6
# # RUN apt-get update && apt-get install -y python3 python3-pip sudo

# RUN useradd -m 2267217f

# RUN chown -R 2267217f:2267217f /home/2267217f

# COPY --chown=2267217f ./src /home/2267217f/app
# COPY --chown=2267217f requirement.txt /home/2267217f/app/requirements.txt

# USER 2267217f

# RUN cd /home/2267217f/app/ && python3.7 -m pip install -r requirements.txt && python3.7 -m pip install dist/*

# WORKDIR /home/2267217f

# # CMD ["python", "./app/experiments/baseline/runner.py", "--test", "false", "--cluster", "true"]

