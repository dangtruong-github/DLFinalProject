import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"


def summary(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss
) -> Tuple[
    float,
    float
]:
    num_correct = 0
    num_samples = 0
    loss_epoch = 0
    loss_avg = 0

    model.eval()

    acc = 0

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            data = data.to(device=device)
            label = label.to(device=device)

            prob = model(data, label)

            pred = torch.argmax(prob, dim=2)

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
                break

        acc = float(num_correct)/float(num_samples) * 100.0
        loss_avg = float(loss_epoch)/float(len(loader))

    return acc, loss_avg
