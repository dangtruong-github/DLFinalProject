from transformers import DataCollatorForSeq2Seq

from data_preprocessing.df_to_hf_dataset import tokenizer, checkpoint


def GetDataCollator(config):
    max_token_length = config["preprocessing"]["max_token_length"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=checkpoint,
                                           padding="max_length",
                                           max_length=max_token_length)

    return data_collator
