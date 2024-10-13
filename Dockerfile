FROM python:3.11-slim

RUN apt-get update
RUN pip install --upgrade pip
RUN apt-get -y install git

ADD training/ ./pokemon_rl/training
ADD PokemonGold.gbc ./pokemon_rl/
ADD PokemonGold_chose_totodile.gbc.state ./pokemon_rl/
ADD requirements.txt ./pokemon_rl/

RUN pip install -r ./pokemon_rl/requirements.txt
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN pip install hnswlib

#CMD ["pwd"]
CMD ["python", "/pokemon_rl/run_training.py"]
#CMD ["sleep", "infinity"]