# Import necessary libraries + config
from configparser import ConfigParser
from data_preprocessing.pipeline import ProcessingPipeline, LoaderPipeline
from train.pipeline import TrainPipeline
from evaluation.pipeline import EvalPipeline
# from evaluation.bleu_score import CalculateBLEUScore

# from train.train import train

config = ConfigParser()
config.read("config.ini")

# Data preprocess + data loader
# !!Create dictionary
ProcessingPipeline(config)
train_loader = LoaderPipeline(config, "train")
val_loader = LoaderPipeline(config, "val")
test_loader = LoaderPipeline(config, "test")

# Train
# train
name_file_save = TrainPipeline(config, train_loader, val_loader)
print(name_file_save)

EvalPipeline(config, test_loader, name_file_save)

# CalculateBLEUScore("I stayed in Peru", "I lived in Peru")

# Evaluation
