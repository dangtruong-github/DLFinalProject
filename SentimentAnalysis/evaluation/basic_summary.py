import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

from evaluation.compute_metrics import compute_metrics
from common_functions.constant import NOATTENTION, ATTENTION
from train.models.encoder_no_attention.init_load_save import initNoAttention
from train.models.encoder_attention.init_load_save import initAttention

device = "cuda" if torch.cuda.is_available() else "cpu"


def Summary(
    config,
    loader: DataLoader,
    model: nn.Module
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
    type_model = config["train"]["model"]

    if type_model == NOATTENTION:
        init_model = initNoAttention
    elif type_model == ATTENTION:
        init_model = initAttention

    _, criterion, _ = init_model(config)

    pred_torch = None
    ref_torch = None

    model.eval()

    acc = 0

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            if bool(config["general"]["test"]):
                if index >= 2:
                    break

            # Data to CUDA if possible
            data = data.to(device=device)
            label = label.to(device=device)
            data = torch.moveaxis(data, 0, 1)

            prob = model(data)

            # print(prob.shape)

            pred = torch.argmax(prob, dim=1)

            current_correct = (pred == label).sum()

            num_correct += current_correct
            num_samples += data.shape[1]

            # print(prob, label)

            loss = criterion(prob, label)

            loss_epoch += loss.item()

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

            # print(f"Summary after model prob shape: {prob.shape}")
            # print(f"Summary after model label shape: {label.shape}")
            # print(f"Summary after model data shape: {data.shape}")
            # print(f"Summary after model pred shape: {pred.shape}")

            loss = criterion(prob, label)

            loss_epoch += loss.item()

            if (index + 1) % 10 == 0:
                # print(f"Finish summary batch {index}")
                if test_bool:
                    break

        acc = float(num_correct)/float(num_samples) * 100.0
        loss_avg = float(loss_epoch)

        # BLEU score
        # pred_torch = pred_torch.numpy()
        # ref_torch = ref_torch.numpy()
        score_total = compute_metrics(config, (pred_torch, ref_torch))

    return acc, loss_avg, score_total
