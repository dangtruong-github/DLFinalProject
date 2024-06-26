import os

from evaluation.basic_summary import Summary
from evaluation.test_model import InferenceTest
from train.train_total import load_model
from train.models.encoder_no_attention.init_load_save import initNoAttention
from train.models.encoder_attention.init_load_save import initAttention
from data_preprocessing.loader import CustomLoaderNew
from common_functions.constant import NOATTENTION, ATTENTION
from common_functions.functions import GetParentPath

test_translation_sentences = [
    "I hate this product",
    "I don't want this one",
    "I really enjoy playing this song"
]


def EvalPipeline(
    config,
    hf_dataset_tokenized,
    name_file_save: str
):
    type_model = config["train"]["model"]

    if type_model == NOATTENTION:
        init_model = initNoAttention
    elif type_model == ATTENTION:
        init_model = initAttention

    parent_directory = GetParentPath(config, __file__)

    SAVE_FOLDER = os.path.join(parent_directory, "model_save", type_model,
                               name_file_save)
    TXT_SAVE_PATH = os.path.join(SAVE_FOLDER, "eval_stats.txt")

    loader = CustomLoaderNew(config, hf_dataset_tokenized, False)

    model, _, optimizer = init_model(config)
    model, _, _ = load_model(model, optimizer, folder=SAVE_FOLDER)
    acc, loss, bleu_score = Summary(config, loader, model)

    with open(TXT_SAVE_PATH, "w", encoding='utf-8') as file:
        file.write("Accuracy of model on test set: {}\n".format(acc))
        file.write("Loss of model on test set: {}\n".format(loss))
        file.write("BLEU Score of model on test set: {}\n".format(bleu_score))

        for sentence in test_translation_sentences:
            file.write("English sentence: {}\n".format(sentence))
            decoded_sentence = InferenceTest(config,
                                             text=sentence,
                                             name_model=name_file_save)
            file.write(f"Translation: {decoded_sentence}\n")
