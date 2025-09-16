FROM python:3.10-slim-bullseye

LABEL maintainer "Robin Opdam"

ENV DB_LOCATION = db_faiss

# update the package lists for upgrades for packages that need upgrading
RUN apt-get update -y

# install build-essential to get a C++ compiler with C++11 support
RUN apt-get install -y build-essential

# set working directory in container
WORKDIR /usr/src/app

# Copy and install packages
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Copy app folder to app folder in container
COPY /app /usr/src/app/

# Expose port for app usage
EXPOSE 8000

# Run locally
CMD ["chainlit", "run", "app.py"]