import evaluate
import numpy as np

from common_functions.functions import GetDict, IndicesToSentence
from data_preprocessing.tokenize import GetTokenizer


def CalculateBLEUScore(
    predictions: list,
    reference: list
) -> float:
    bleu = evaluate.load('bleu')

    # Compute the BLEU score
    results = bleu.compute(predictions=predictions,
                           references=reference)

    # Print the BLEU score
    print(f"BLEU score: {results['bleu']:.4f}")
    return results['bleu']


def BLEUScoreFromIndices(
    config,
    predictions: np.array,
    references: np.array
) -> float:
    """
    Args:
    - config: config
    - predictions and reference: two np.array of
                                shape (total_sentence, total_indices)
    Return:
    - bleu_score: bleu score
    """
    if predictions.shape != references.shape:
        raise ValueError(f"Predictions shape {predictions.shape}"
                         "is not equal to references shape {references.shape}")
    _, en_dict = GetDict(config)

    pred_sentence = []
    ref_sentence = []

    print("<sos> token index: {}".format(en_dict["<sos>"]))
    print("<eos> token index: {}".format(en_dict["<eos>"]))

    for i in range(predictions.shape[0]):
        pred_sentence.append(IndicesToSentence(predictions[i, :], en_dict))
        # print(predictions[i, :])
        ref_sentence.append([IndicesToSentence(references[i, :], en_dict)])

    # Load the BLEU metric
    print(f"Pred sentence inside BLEUSCore func: {pred_sentence}")
    print(f"Ref sentence inside BLEUSCore func: {ref_sentence}")
    score = CalculateBLEUScore(pred_sentence, ref_sentence)

    return score


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(config, eval_preds):
    tokenizer = GetTokenizer(config)
    metric = evaluate.load("sacrebleu")

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)

    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in preds]

    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
