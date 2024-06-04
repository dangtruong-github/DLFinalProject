import torch

import os

from evaluation.basic_summary import summary
from train.models.rnn_seq2seq.init_load_save import initSeq2Seq
from common_functions.constant import SEQ2SEQ, TRANSFORMER

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, map_location=torch.device(device))

    print(type(checkpoint["model_state_dict"]))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch


def train(
    train_loader,
    val_loader,
    source_dict,
    target_dict,
    config
):
    train_acc_list = []
    train_loss_list = []

    val_acc_list = []
    val_loss_list = []

    num_epochs = config["train"]["epoch"]
    type_model = config["train"]["type_model"]
    file_save = config["train"]["file_save"]
    batch_print = config["train"]["batch_print"]

    if type_model == SEQ2SEQ:
        init_model = initSeq2Seq
    elif type_model == TRANSFORMER:
        pass

    cur_epoch = -1

    model, criterion, optimizer = init_model(config, source_dict, target_dict)

    # numpy_final_result = [[] for _ in range(20)]

    SAVE_FOLDER = os.path.join(os.getcwd(), "model_save", type_model)
    MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, "{}.pt".format(file_save))
    # JSON_SAVE_PATH = os.path.join(SAVE_FOLDER, "{}.json".format(file_save))

    if os.path.exists(MODEL_SAVE_PATH):
        model, optimizer, cur_epoch = load_model(model,
                                                 optimizer,
                                                 path=MODEL_SAVE_PATH)

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

        for batch_idx, (data, label) in enumerate(train_loader):
            # Data to CUDA if possible
            data = data.to(device=device)
            label = label.to(device=device)
            print(data.shape)
            print(label.shape)

            data = torch.moveaxis(data, 1, 0)
            label = torch.moveaxis(label, 1, 0)
            print(data.shape)
            print(label.shape)

            optimizer.zero_grad()

            prob = model(data, label)
            # prob.requires_grad=True
            prob.retain_grad()

            pred = torch.argmax(prob, dim=2)

            current_correct = (pred == label).sum()
            current_size = pred.shape[0] * pred.shape[1]

            correct_samples += current_correct
            total_samples += current_size

            prob = torch.moveaxis(prob, (1, 2), (0, 1))
            label = torch.moveaxis(label, 1, 0)

            # print(data.shape)
            # print(label.shape)
            # print(pred.shape)
            # print(prob.shape)

            loss = criterion(prob, label)
            loss.retain_grad()
            # loss.requires_grad=True
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            # optimizer.requires_grad=True
            optimizer.step()

            loss_epoch += loss.item()

            if batch_idx % batch_print == batch_print - 1:
                curr_acc = float(current_correct) / float(current_size) * 100.0
                print(f"Batch {batch_idx + 1}: Accuracy: {curr_acc}")
                print(f"Loss: {float(loss.item()) / float(pred.shape[1])}")
                save_model(model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           path=MODEL_SAVE_PATH)

        # Validation
        val_acc, val_loss = summary(val_loader, model, criterion)

        train_acc_cur = float(correct_samples) / float(total_samples + 1e-12)
        train_acc_cur *= 100

        train_acc_list.append(train_acc_cur)
        train_loss_list.append(float(loss_epoch) / float(len(train_loader)))

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        # for i in range(20):
        #    numpy_final_result[i].extend(final_result[i])
        #    print(f"Prob for {i + 1}: min {np.min(numpy_final_result[i])},
        # max: {np.max(numpy_final_result[i])}")

        if epoch % 1 == 0:
            save_model(model=model,
                       optimizer=optimizer,
                       epoch=epoch,
                       path=MODEL_SAVE_PATH)

        cur_epoch = epoch

        print(f"Epoch {epoch + 1}:")

        print(f"Train accuracy: {train_acc_list[-1]}%")
        print(f"Train loss: {train_loss_list[-1]}")

        print(f"Val accuracy: {val_acc_list[-1]}%")
        print(f"Val loss: {val_loss_list[-1]}")
