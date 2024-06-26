"""import numpy as np

train_new = np.load("./data/train_indices.npy")
train_all = np.load("./data/train_indices_old.npy")
train_all = train_all[:train_new.shape[0], :]

indices_wrong = []

for i in range(train_new.shape[0]):
    equals = np.array_equal(train_new[i], train_all[i])

    if not equals:
        indices_wrong.append(i)
        print(f"Index wrong: {i}")
        print(f"New array: {train_new[i]}")
        print(f"Old array: {train_all[i]}")

print(f"Indices wrong: {indices_wrong}")


import configparser

# Create a ConfigParser instance
config = configparser.ConfigParser()

# Read your configuration file (if needed)
config.read("./config.ini")

# Check if a section and key exist
section_name = "preprocessing"
key_name = "train_sample_used"

if config.has_section(section_name) and config.has_option(section_name,
                                                          key_name):
    # The section and key both exist
    value = config.get(section_name, key_name)
    print(f"The value for {key_name} in {section_name} is: {value}")
else:
    print(f"{key_name} does not exist in {section_name}.")
"""

from data_preprocessing.tokenize import GetTokenizer
import configparser

# Create a ConfigParser instance
config = configparser.ConfigParser()

# Read your configuration file (if needed)
config.read("./config.ini")

tokenizer = GetTokenizer(config)

print(tokenizer.pad_token_id)
