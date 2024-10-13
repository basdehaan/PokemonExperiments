FROM python:3.11-slim

RUN apt-get update
RUN pip install --upgrade pip
RUN apt-get -y install git

ADD PokemonGold.gbc /pokemon_rl/
ADD PokemonGold_chose_totodile.gbc.state /pokemon_rl/
ADD requirements.txt /pokemon_rl/

RUN pip install -r /pokemon_rl/requirements.txt
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN pip install hnswlib

ADD training/ ./pokemon_rl/training
RUN mkdir /pokemon_rl/training/_session_continuous

WORKDIR /pokemon_rl/training
CMD ["python", "run_training.py"]

# TODO: save model when training reaches x iterations
# TODO: command to extract trained model from the container - volume?