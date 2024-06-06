import evaluate
import numpy as np

from common_functions.functions import GetDict, IndicesToSentence


def CalculateBLEUScore(
    predictions: str,
    reference: str
) -> float:
    pred_sentences = [predictions]
    ref_sentences = [[reference]]

    bleu = evaluate.load('bleu')

    # Compute the BLEU score
    results = bleu.compute(predictions=pred_sentences,
                           references=ref_sentences)

    # Print the BLEU score
    print(f"BLEU score: {results['bleu']:.4f}")
    return results['bleu']


def BLEUScoreFromIndices(
    config,
    predictions: np.array,
    reference: np.array
) -> float:
    _, en_dict = GetDict(config)

    pred_sentence = IndicesToSentence(predictions, en_dict)
    ref_sentence = IndicesToSentence(reference, en_dict)

    # Load the BLEU metric
    score = CalculateBLEUScore(pred_sentence, ref_sentence)

    return score
