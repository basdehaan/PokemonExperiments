FROM python3.10

ADD ./baselines .

CMD ["python3.10", "./baselines/run_training.py"]
