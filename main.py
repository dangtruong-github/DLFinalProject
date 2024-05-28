# Import necessary libraries + config
from configparser import ConfigParser

from train.train import train

config = ConfigParser()
config.read("config.ini")

# Data preprocess + data loader

# Train
train

# Evaluation
