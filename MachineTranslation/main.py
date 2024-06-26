# Import necessary libraries + config
from configparser import ConfigParser
import os

from data_preprocessing.pipeline import ProcessingPipeline
from train.pipeline import TrainPipeline
from evaluation.pipeline import EvalPipeline
# from evaluation.bleu_score import CalculateBLEUScore

# from train.train import train

config = ConfigParser()
config_path = os.path.dirname(__file__)
config.read(os.path.join(config_path, "config.ini"))

# Data preprocess + data loader
# !!Create dictionary
hf_train_tokenized = ProcessingPipeline(config, "train", save=True)
hf_val_tokenized = ProcessingPipeline(config, "val", save=True)
hf_test_tokenized = ProcessingPipeline(config, "test", save=True)

# Train
# train
name_file_save = TrainPipeline(config, hf_train_tokenized, hf_val_tokenized)
print(name_file_save)

EvalPipeline(config, hf_test_tokenized, name_file_save)

# CalculateBLEUScore("I stayed in Peru", "I lived in Peru")

# Evaluation
