import numpy as np

train_new = np.load("./data/train_indices.npy")
train_all = np.load("./data/train_indices_old.npy")
train_all = train_all[:train_new.shape[0], :]

indices_wrong = []

for i in range(train_new.shape[0]):
    equals = np.array_equal(train_new[i], train_all[i])

    if not equals:
        indices_wrong.append(i)
        print(f"Index wrong: {i}")
        print(f"New array: {train_new[i]}")
        print(f"Old array: {train_all[i]}")

print(f"Indices wrong: {indices_wrong}")
