from torch.utils.data import DataLoader

from train.train_total import train


def TrainPipeline(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> str:
    name_file_save = train(config, train_loader, val_loader)
    return name_file_save
