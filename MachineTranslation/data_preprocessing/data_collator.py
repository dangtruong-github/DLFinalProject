from transformers import DataCollatorForSeq2Seq

from data_preprocessing.tokenize import GetTokenizer, checkpoint


def GetDataCollator(config):
    max_token_length = int(config["preprocessing"]["max_token_length"])

    tokenizer = GetTokenizer(config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=checkpoint,
                                           padding="max_length",
                                           max_length=max_token_length)

    return data_collator
