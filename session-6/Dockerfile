# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /first_image

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --quiet -r requirements.txt

# Copy code to the working directory
COPY . /first_image

# command to run on container start
ENTRYPOINT ["python","./src/entrypoint.py"]
