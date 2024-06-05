# using library to implement
# https://pypi.org/project/evaluate/

def bleu_score(
    label: list,
    predict: list
) -> float:
    #there may be several references
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([label], predict)

    return BLEUscore