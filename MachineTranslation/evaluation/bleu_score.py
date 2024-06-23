import evaluate
import numpy as np

from common_functions.functions import GetDict, IndicesToSentence


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
