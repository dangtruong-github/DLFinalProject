import os

from evaluation.basic_summary import Summary
from evaluation.test_model import InferenceTest
from train.train_total import load_model
from train.models.rnn_seq2seq.init_load_save import initSeq2Seq
from train.models.transformer.init_load_save import initTransformer
from data_preprocessing.loader import CustomLoaderNew
from common_functions.constant import SEQ2SEQ, TRANSFORMER
from common_functions.functions import GetParentPath

test_translation_sentences = [
    "Hello, how are you?",
    "I live in Hanoi",
    "I am eight years old"
]


def EvalPipeline(
    config,
    hf_dataset_tokenized,
    name_file_save: str
):
    type_model = config["train"]["model"]

    if type_model == SEQ2SEQ:
        init_model = initSeq2Seq
    elif type_model == TRANSFORMER:
        init_model = initTransformer

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
