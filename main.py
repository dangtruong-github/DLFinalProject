# Import necessary libraries + config
from configparser import ConfigParser
from data_preprocessing.pipeline import ProcessingPipeLine

# from train.train import train

config = ConfigParser()
config.read("config.ini")

# Data preprocess + data loader
# !!Create dictionary
ProcessingPipeLine(config)

# Train
# train

# Evaluation
