FROM python:3.11-slim

RUN apt-get update
RUN pip install --upgrade pip

ADD pokemon_rl/ ./pokemon_rl/
ADD requirements.txt ./pokemon_rl/

RUN pip install -r ./pokemon_rl/requirements.txt

#CMD ["pwd"]
CMD ["python", "/pokemon_rl/run_training.py"]
#CMD ["sleep", "infinity"]