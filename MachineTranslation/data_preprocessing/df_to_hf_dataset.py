from datasets import Dataset


def DfToHfDataset(config, df):
    hf_dataset = Dataset.from_dict({
        "english": df["english"],
        "vietnamese": df["vietnamese"]
    })

    return hf_dataset
