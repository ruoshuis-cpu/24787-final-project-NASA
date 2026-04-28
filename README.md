# NASA C-MAPSS Turbofan Engine Degradation Prediction

This project predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. The project compares two sequence-based deep learning models:

- LSTM baseline model
- Deep CNN (DCNN) model variant

The task is to estimate engine RUL from multivariate sensor time-series data.

---

## Environment Setup

This project was developed in Python using PyTorch.

Recommended Python version:

```bash
Python 3.9+
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

Required packages:

```bash
numpy
pandas
matplotlib
scikit-learn
torch
jupyter
```

---

## Dataset Download

Download the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset from the NASA Prognostics Data Repository:

https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

After downloading, extract the dataset and place the data folder in the project directory.

The code expects the FD001 files:

```bash
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Example project structure:

```bash
24787-final-project-NASA/
├── README.md
├── NASA_C_MAPSS_Turbofan_Engine_Degradation.ipynb
├── reproduce_results.ipynb
├── checkpoints/
│   ├── lstm_model.pt
│   └── dcnn_model.pt
└── CMAPSSData/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

If the dataset is stored somewhere else, update the dataset path in the notebook.

---

## Files in This Repository

```bash
NASA_C_MAPSS_Turbofan_Engine_Degradation.ipynb
```

Main training notebook. It includes data loading, preprocessing, model definitions, model training, evaluation, and result visualization.

```bash
reproduce_results.ipynb
```

Reproduction notebook. It loads the saved model checkpoints and regenerates the main evaluation metrics and figures without retraining from scratch.

```bash
checkpoints/
```

Contains trained PyTorch model checkpoints:

```bash
lstm_model.pt
dcnn_model.pt
```

---

## How to Train the Models

Open the main notebook:

```bash
jupyter notebook NASA_C_MAPSS_Turbofan_Engine_Degradation.ipynb
```

Run all cells.

The notebook trains:

1. LSTM baseline model
2. DCNN model variant

After training, the model checkpoints are saved to:

```bash
checkpoints/lstm_model.pt
checkpoints/dcnn_model.pt
```

---

## How to Reproduce Key Results

To reproduce the key results from the report without retraining:

1. Make sure the dataset is downloaded and placed in the expected folder.
2. Make sure the saved checkpoints are available in the `checkpoints/` folder.
3. Open the reproduction notebook:

```bash
jupyter notebook reproduce_results.ipynb
```

4. Run all cells.

The reproduction notebook will:

- Load the FD001 test data
- Load the saved LSTM and DCNN checkpoints
- Evaluate both models on the test set
- Regenerate the reported metrics
- Regenerate the key comparison plots

The main reported metrics are:

- Test RMSE
- PHM score

---

## Model Summary

### LSTM Baseline

The LSTM model processes sensor readings as time-series sequences and predicts the RUL from the final hidden representation.

### DCNN Variant

The DCNN model applies temporal convolution layers over the input sensor sequence to learn local degradation patterns across time.

---

## Notes

The default configuration uses:

```python
subset = "FD001"
window_size = 30
rul_clip = 125
batch_size = 128
num_epochs = 100
```

The notebook automatically uses GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If no GPU is available, the code runs on CPU.

---

## Author

Ruoshui Song
Hongfei Liu
