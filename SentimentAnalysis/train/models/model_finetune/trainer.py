from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch

from train.models.model_finetune.loss_callback import LossHistoryCallback
from train.models.model_finetune.model import GetFineTuneModel
from data_preprocessing.tokenize import GetTokenizer
from data_preprocessing.data_collator import GetDataCollator
from evaluation.compute_metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"


def CreateTrainer(config, hf_train_tokenized, hf_val_tokenized):
    loss_history = LossHistoryCallback()

    learning_rate = float(config["train"]["learning_rate"])
    batch_size = int(config["train"]["batch_size"])
    weight_decay = float(config["train"]["weight_decay"])
    epochs = int(config["train"]["epoch"])

    output_dir = "./eval"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        num_train_epochs=epochs,
        predict_with_generate=True,
        report_to=None,
        fp16=(device == "cuda"),
        logging_steps=1
    )

    tokenizer = GetTokenizer(config)
    data_collator = GetDataCollator(config)
    model = GetFineTuneModel(config)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_tokenized,
        eval_dataset=hf_val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_history],
    )

    return trainer, loss_history, output_dir
