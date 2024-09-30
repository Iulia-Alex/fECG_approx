import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import os
from complex_data import SpectrogramDataset
from complex_network import ComplexUNet

import math
from torch.utils.data.dataset import Subset
from torch import default_generator, randperm
from torch._utils import _accumulate



def complex_mse_loss(output, target):
    return (0.5*(output - target)**2).mean(dtype=torch.complex64)


class ComplexMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_pred):
        y_real, y_imag = torch.real(y), torch.imag(y)
        y_pred_real, y_pred_imag = torch.real(y_pred), torch.imag(y_pred)
        
        return torch.mean((y_real - y_pred_real) ** 2 + (y_imag - y_pred_imag) ** 2)

def random_split(dataset, lengths,
                 generator=default_generator):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 15
    epochs = 35
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join("../Licenta/unet", "ecg")
    best_loss = 99

    model_save_path = os.path.join(current_dir, "models", "complex_unet2.pth")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(torch.cuda.is_available())

    train_dataset = SpectrogramDataset(data_path)

    random_gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator = random_gen)

    # TODO:  create sample child with few indexes ----> search on GPT :)
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  
                    )
    val_dataloader = DataLoader(dataset = val_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                    )

    model = ComplexUNet(128 * 128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=1e-5)
    # optimizer = optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # criterion = nn.BCEWithLogitsLoss() #MSE
    criterion = ComplexMSE()
    # criterion = complex_mse_loss()

    show_epoch = 1
    history = {"train_loss": [], "val_loss":[]}
    for epoch in tqdm(range(epochs), desc = f"Total Epochs: {epochs}"):
        model.train()
        train_running_loss = 0
        for idx, img_and_target in enumerate(tqdm(train_dataloader, desc = f"Epoch {show_epoch} of {epochs}")):
            img = img_and_target[0].to(device)
            target = img_and_target[1].to(device)

            pred = model(img)
            # print(pred.shape)
            # pred_l = torch.tensor(torch.stack((pred.real, pred.imag)), dtype=torch.float64)
            # target_l = torch.tensor(torch.stack((target.real, target.imag)), dtype=torch.float64)

            loss = criterion(pred, target)
            # loss = criterion(pred, target)
            # loss = complex_mse_loss(pred,target)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            # print(loss)
            loss.backward()
            # torch.abs(loss).backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        history["train_loss"].append(train_loss)
        
        # Start evaluation
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_and_target in enumerate(tqdm(val_dataloader)):
                img = img_and_target[0].to(device)
                target = img_and_target[1].to(device)

                pred = model(img)

                loss = criterion(pred, target)
                val_running_loss += loss.item()

            val_loss = train_running_loss / (idx + 1)
            history["val_loss"].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

        print()
        print(f"\nEpoch {show_epoch} Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print()
        show_epoch += 1

    # torch.save(model.state_dict(), model_save_path)
    print(history)