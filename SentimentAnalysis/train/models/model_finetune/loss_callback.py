from transformers import TrainerCallback


class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_bleu = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(logs)
            if "loss" in logs:
                self.train_loss.append(logs["loss"])
                self.epochs.append(int(logs["epoch"]))
            if "eval_loss" in logs:
                self.eval_loss.append(logs["eval_loss"])
            if "eval_bleu" in logs:
                self.eval_bleu.append(logs["eval_bleu"])

            print(self.train_loss)
            print(self.epochs)
            print(self.eval_loss)

# Assuming you have these objects already
# model, tokenizer, train_dataset, eval_dataset
