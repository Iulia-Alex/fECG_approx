# Fetal ECG Approximation
This branch is dedicated to the development of a fetal ECG approximation algorithm, using complex neural networks.

## Folder Structure
- `data/`: Contains the dataset used for training and testing the model.
- `models/`: Contains the trained models.
- `results/`: Contains various results obtained during testing.
- `src/`: Contains the source code for the project.

## Usage
1. Clone the repository.
2. Acquire the required dependencies, including packages and datasets. See `env.yml` for the required packages.
3. Run the `train.py` script to train the model.

Disclaimer: Average time for an epoch on GTX 3060 is 2 minutes.

## Next Steps
- [ ] Try different architectures, including:
    - [ ] A single kernel / diag for each layer.
    - [ ] Concat instead of sum in residual connections.
    - [ ] Multiple layers in each block.
    - [ ] SE Complex Conv ?
    - [ ] Complex attention ?

- [ ] Add more activation functions.


## Contributors
- [Iulia Orvas](https://github.com/Iulia-Alex)
- [Andrei Radu](https://github.com/andrei-radu)
