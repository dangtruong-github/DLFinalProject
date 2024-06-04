# Import necessary libraries + config
from configparser import ConfigParser
from data_preprocessing.pipeline import ProcessingPipeline

# from train.train import train

config = ConfigParser()
config.read("config.ini")

# Data preprocess + data loader
# !!Create dictionary
ProcessingPipeline(config)

# Train
# train

# Evaluation
