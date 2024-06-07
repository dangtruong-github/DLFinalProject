import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

from evaluation.bleu_score import BLEUScoreFromIndices

device = "cuda" if torch.cuda.is_available() else "cpu"


def Summary(
    config,
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss
) -> Tuple[
    float,
    float,
    float
]:
    num_correct = 0
    num_samples = 0
    loss_epoch = 0
    loss_avg = 0

    test_bool = bool(config["general"]["test"])

    pred_torch = None
    ref_torch = None

    model.eval()

    acc = 0

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            if bool(config["general"]["test"]):
                if index >= 2:
                    break

            data = data.to(device=device)
            label = label.to(device=device)

            prob = model(data, label)

            pred = torch.argmax(prob, dim=2)

            if index == 0:
                pred_torch = pred
                ref_torch = label
                # print(f"Summary prediction shape: {pred_torch.shape}")
                # print(f"Summary label shape: {ref_torch.shape}")
            else:
                pred_torch = torch.cat([pred_torch, pred], axis=0)
                ref_torch = torch.cat([ref_torch, label], axis=0)
                # print(f"Summary prediction shape: {pred_torch.shape}")
                # print(f"Summary label shape: {ref_torch.shape}")

            current_correct = (pred == label).sum()
            current_size = pred.shape[0] * pred.shape[1]

            num_correct += current_correct
            num_samples += current_size

            prob = torch.moveaxis(prob, (1, 2), (0, 1))
            label = torch.moveaxis(label, 1, 0)

            # print(data.shape)
            # print(label.shape)
            # print(pred.shape)

            loss = criterion(prob, label)

            loss_epoch += loss.item()

            if (index + 1) % 100 == 0:
                print(f"Finish summary batch {index}")
                if test_bool:
                    break

        acc = float(num_correct)/float(num_samples) * 100.0
        loss_avg = float(loss_epoch)/float(len(loader))

        # BLEU score
        pred_torch = pred_torch.numpy()
        ref_torch = ref_torch.numpy()
        bleu_score = BLEUScoreFromIndices(config, pred_torch, ref_torch)

    return acc, loss_avg, bleu_score
