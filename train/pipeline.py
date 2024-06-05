from torch.utils.data import DataLoader

from .train_total import train


def TrainPipeline(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader
):
    train(config, train_loader, val_loader)
