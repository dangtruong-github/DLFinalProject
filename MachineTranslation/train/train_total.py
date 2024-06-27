import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
from typing import Tuple
from datetime import datetime

from evaluation.basic_summary import Summary
from evaluation.bleu_score import compute_metrics
from train.models.rnn_seq2seq.init_load_save import initSeq2Seq
from train.models.transformer.init_load_save import initTransformer
from train.models.model_finetune.trainer import CreateTrainer
from data_preprocessing.loader import CustomLoaderNew
from common_functions.constant import SEQ2SEQ, TRANSFORMER
from common_functions.functions import GetParentPath

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_model(
    model: nn.Module,
    optimizer,
    epoch: int,
    folder: str
):
    final_path = os.path.join(folder, "model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)


def load_model(
    model: nn.Module,
    optimizer,
    folder: str
) -> Tuple[
    nn.Module,
    optim.Optimizer,
    int
]:
    path = os.path.join(folder, "model.pt")
    checkpoint = torch.load(path, map_location=torch.device(device))

    # print(type(checkpoint["model_state_dict"]))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def FinetuneTrain(
    config,
    hf_train_tokenized,
    hf_val_tokenized
) -> str:
    trainer, loss_history, output_dir = CreateTrainer(config,
                                                      hf_train_tokenized,
                                                      hf_val_tokenized)
    trainer.train()

    print(f"Epoch: {loss_history.epochs}")
    print(f"Train loss: {loss_history.train_loss}")
    print(f"Eval loss: {loss_history.eval_loss}")
    print(f"Eval acc: {loss_history.eval_bleu}")

    return output_dir


def train(
    config,
    hf_train_tokenized,
    hf_val_tokenized
) -> str:
    train_acc_list = []
    train_loss_list = []
    train_bleu_score_list = []

    val_acc_list = []
    val_loss_list = []
    val_bleu_score_list = []

    type_model = config["train"]["model"]

    if type_model == SEQ2SEQ:
        init_model = initSeq2Seq
    elif type_model == TRANSFORMER:
        init_model = initTransformer

    train_loader = CustomLoaderNew(config, hf_train_tokenized, True)
    val_loader = CustomLoaderNew(config, hf_val_tokenized, False)

    batch_print = int(config["train"]["batch_print"])

    test_bool = bool(config["general"]["test"])

    num_epochs = int(config["train"]["epoch"])
    if test_bool:
        num_epochs = int(config["train"]["epoch_test"])

    parent_directory = GetParentPath(config, __file__)

    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    # print(cur_date)
    file_save = "{}_{}".format(type_model, cur_date)

    cur_epoch = -1

    model, criterion, optimizer = init_model(config)

    # numpy_final_result = [[] for _ in range(20)]

    SAVE_FOLDER = os.path.join(parent_directory, "model_save", type_model,
                               file_save)

    if not os.path.exists(os.path.dirname(SAVE_FOLDER)):
        os.makedirs(os.path.dirname(SAVE_FOLDER))

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, "model.pt")
    JSON_SAVE_PATH = os.path.join(SAVE_FOLDER, "train_stats.json")

    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(JSON_SAVE_PATH):
        model, optimizer, cur_epoch = load_model(model,
                                                 optimizer,
                                                 folder=SAVE_FOLDER)

        with open(JSON_SAVE_PATH, "r") as f:
            json_data = json.load(f)

            train_acc_list = json_data["train_acc_list"]
            train_loss_list = json_data["train_loss_list"]
            train_bleu_score_list = json_data["train_bleu_score"]

            val_acc_list = json_data["val_acc_list"]
            val_loss_list = json_data["val_loss_list"]
            val_bleu_score_list = json_data["val_bleu_score"]

        # with open(NUMPY_SAVE_PATH, 'rb') as f:
        #    numpy_final_result = pickle.load(f)

    for epoch in range(num_epochs):
        if cur_epoch >= epoch:
            continue

        correct_samples = 0
        total_samples = 0

        loss_epoch = 0

        print("----------------------------------------")

        model.train()

        pred_torch = None
        ref_torch = None

        for batch_idx, (data, label) in enumerate(train_loader):
            if test_bool:
                if batch_idx >= 2:
                    break

            # Data to CUDA if possible
            data = data.to(device=device)
            label = label.to(device=device)
            # print(f"Orig data shape: {data.shape}")
            # print(f"Orig label shape: {label.shape}")

            data = torch.moveaxis(data, 1, 0)
            label = torch.moveaxis(label, 1, 0)
            # print(f"After data shape: {data.shape}")
            # print(f"After label shape: {label.shape}")

            optimizer.zero_grad()

            prob = model(data, label)
            prob.retain_grad()

            pred = torch.argmax(prob, dim=2)

            current_correct = (pred == label).sum()
            current_size = pred.shape[0] * pred.shape[1]

            correct_samples += current_correct
            total_samples += current_size

            prob = torch.moveaxis(prob, (1, 2), (0, 1))
            label = torch.moveaxis(label, 1, 0)
            pred = torch.moveaxis(pred, 1, 0)

            if batch_idx == 0:
                pred_torch = pred
                ref_torch = label
                print(f"Train prediction shape: {pred_torch.shape}")
                print(f"Train label shape: {ref_torch.shape}")
            else:
                pred_torch = torch.cat([pred_torch, pred], axis=0)
                ref_torch = torch.cat([ref_torch, label], axis=0)
                print(f"Train prediction shape: {pred_torch.shape}")
                print(f"Train label shape: {ref_torch.shape}")

            # print(f"After model prob shape: {prob.shape}")
            # print(f"After model label shape: {label.shape}")
            # print(f"After model data shape: {data.shape}")
            # print(f"After model pred shape: {pred.shape}")

            loss = criterion(prob, label)
            loss.retain_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            loss_epoch += loss.item()

            if batch_idx % batch_print == batch_print - 1:
                curr_acc = float(current_correct) / float(current_size) * 100.0
                print(f"Batch {batch_idx + 1}: Accuracy: {curr_acc}")
                print(f"Loss: {loss.item()}")

        # BLEU score
        # pred_torch = pred_torch.numpy()
        # ref_torch = ref_torch.numpy()
        train_bleu_score = compute_metrics(config, (pred_torch, ref_torch))
        pred_torch = None
        ref_torch = None

        # Validation
        val_acc, val_loss, val_bleu_score = Summary(config, val_loader, model)

        print("Finish summary")

        train_acc_cur = float(correct_samples) / float(total_samples + 1e-12)
        train_acc_cur *= 100

        train_acc_list.append(train_acc_cur)
        train_loss_list.append(float(loss_epoch))
        train_bleu_score_list.append(train_bleu_score)

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        val_bleu_score_list.append(val_bleu_score)

        prev_bleu_score = 0
        if len(val_bleu_score_list) >= 2:
            prev_bleu_score = val_bleu_score_list[-2]["bleu"]

        cur_bleu_score = val_bleu_score["bleu"]
        # for i in range(20):
        #    numpy_final_result[i].extend(final_result[i])
        #    print(f"Prob for {i + 1}: min {np.min(numpy_final_result[i])},
        # max: {np.max(numpy_final_result[i])}")

        if epoch % 1 == 0 and prev_bleu_score < cur_bleu_score:
            save_model(model=model,
                       optimizer=optimizer,
                       epoch=epoch,
                       folder=SAVE_FOLDER)

            with open(JSON_SAVE_PATH, "w") as f:
                json_data_save = {
                    "train_acc_list": train_acc_list,
                    "train_loss_list": train_loss_list,
                    "train_bleu_score": train_bleu_score_list,

                    "val_acc_list": val_acc_list,
                    "val_loss_list": val_loss_list,
                    "val_bleu_score": val_bleu_score_list
                }

                json.dump(json_data_save, f)

        cur_epoch = epoch

        print(f"Epoch {epoch + 1}:")

        print(f"Train accuracy: {train_acc_list[-1]}%")
        print(f"Train loss: {train_loss_list[-1]}")
        print(f"Train BLEU score: {train_bleu_score_list[-1]['bleu']}")

        print(f"Val accuracy: {val_acc_list[-1]}%")
        print(f"Val loss: {val_loss_list[-1]}")
        print(f"Val BLEU score: {val_bleu_score_list[-1]['bleu']}")

    return file_save
