from torch.utils.data import DataLoader

import os

from evaluation.basic_summary import Summary
from train.train_total import load_model
from train.models.rnn_seq2seq.init_load_save import initSeq2Seq
from train.models.transformer.init_load_save import initTransformer
from common_functions.constant import SEQ2SEQ, TRANSFORMER
from common_functions.functions import GetParentPath


def EvalPipeline(
    config,
    loader: DataLoader,
    name_file_save: str
):
    type_model = config["train"]["model"]
    parent_folder_name = config["general"]["containing_folder"]

    parent_directory = GetParentPath(parent_folder_name, __file__)

    SAVE_FOLDER = os.path.join(parent_directory, "model_save", type_model)
    MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, "{}.pt".format(name_file_save))
    print(MODEL_SAVE_PATH)
    TXT_SAVE_PATH = os.path.join(SAVE_FOLDER, "{}.txt".format(name_file_save))

    if type_model == SEQ2SEQ:
        init_model = initSeq2Seq
    elif type_model == TRANSFORMER:
        init_model = initTransformer

    model, _, optimizer = init_model(config)
    model, _, _ = load_model(model, optimizer, path=MODEL_SAVE_PATH)
    acc, loss, bleu_score = Summary(config, loader, model)

    with open(TXT_SAVE_PATH, "w") as file:
        file.write("Accuracy of model on test set: {}\n".format(acc))
        file.write("Loss of model on test set: {}\n".format(loss))
        file.write("BLEU Score of model on test set: {}".format(bleu_score))
