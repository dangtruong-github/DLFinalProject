# Import necessary libraries + config
from configparser import ConfigParser
from data_preprocessing.pipeline import ProcessingPipeline, LoaderPipeline
from train.pipeline import TrainPipeline
# from evaluation.bleu_score import CalculateBLEUScore

# from train.train import train

config = ConfigParser()
config.read("config.ini")

# Data preprocess + data loader
# !!Create dictionary
ProcessingPipeline(config)
train_loader = LoaderPipeline(config, "train")
val_loader = LoaderPipeline(config, "val")

# Train
# train
TrainPipeline(config, train_loader, val_loader)

# CalculateBLEUScore("I stayed in Peru", "I lived in Peru")

# Evaluation
