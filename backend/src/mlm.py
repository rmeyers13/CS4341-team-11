# this should let gpu acceleration run
import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def mcc_multioutput(y_true, y_pred):
    """
    Compute MCC for each output column and return the mean.
    Works with y_true, y_pred of shape (n_samples, n_outputs).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If 1D, just use normal MCC
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        return matthews_corrcoef(y_true.ravel(), y_pred.ravel())

    mccs = []
    for i in range(y_true.shape[1]):
        mccs.append(matthews_corrcoef(y_true[:, i], y_pred[:, i]))
    return float(np.mean(mccs))


# =====================
#  Dataset / preprocessing
# =====================

def getDataset(filename):
    # defines the column names, and then defines the column names we want
    names = ["LightLevel", "Weather", "RoadCondition", "Longitude", "Latitude"]
    colsToUse = ["LightLevel", "Weather", "RoadCondition", "Longitude", "Latitude"]

    # creates the dataset
    dataset = (
        pd.read_csv(filename, names=names, skiprows=1, usecols=colsToUse)
        .dropna()
        .reset_index(drop=True)
    )
    return dataset


def scaleData(dataset):
    # Text to Number Map
    lightLevelMap = {
        'Not reported': 0, 'Other': 0, 'Reported but invalid': 0, 'Unknown': 0,
        'Dawn': 1, 'Dusk': 1,
        'Daylight': 2,
        'Dark - lighted roadway': 3, 'Dark - roadway not lighted': 3,
        'Dark - unknown roadway lighting': 3
    }

    # Note that blowing sand is blowing sand or snow
    weatherMap = {'Blowing sand/snow': 1,
                  'Blowing sand/snow/Blowing sand/snow': 1,
                  'Blowing sand/snow/Clear': 1,
                  'Blowing sand/snow/Cloudy': 1,
                  'Blowing sand/snow/Fog/smog/smoke': 1,
                  'Blowing sand/snow/Other': 1,
                  'Blowing sand/snow/Rain': 1,
                  'Blowing sand/snow/Severe crosswinds': 2,
                  'Blowing sand/snow/Sleet/hail (freezing rain or drizzle)': 1,
                  'Blowing sand/snow/Snow': 1,
                  'Blowing sand/snow/Unknown': 1,

                  'Clear': 3,
                  'Clear/Blowing sand/snow': 1,
                  'Clear/Clear': 3,
                  'Clear/Cloudy': 4,
                  'Clear/Fog/smog/smoke': 5,
                  'Clear/Other': 3,
                  'Clear/Rain': 6,
                  'Clear/Reported but invalid': 3,
                  'Clear/Severe crosswinds': 7,
                  'Clear/Sleet/hail (freezing rain or drizzle)': 8,
                  'Clear/Snow': 9,
                  'Clear/Unknown': 3,

                  'Cloudy': 4,
                  'Cloudy/Blowing sand/snow': 1,
                  'Cloudy/Clear': 4,
                  'Cloudy/Cloudy': 4,
                  'Cloudy/Fog/smog/smoke': 5,
                  'Cloudy/Other': 4,
                  'Cloudy/Rain': 6,
                  'Cloudy/Reported but invalid': 4,
                  'Cloudy/Severe crosswinds': 7,
                  'Cloudy/Sleet/hail (freezing rain or drizzle)': 8,
                  'Cloudy/Snow': 9,
                  'Cloudy/Unknown': 4,

                  'Fog/smog/smoke': 5,
                  'Fog/smog/smoke/Clear': 5,
                  'Fog/smog/smoke/Cloudy': 5,
                  'Fog/smog/smoke/Fog/smog/smoke': 5,
                  'Fog/smog/smoke/Other': 5,
                  'Fog/smog/smoke/Rain': 6,
                  'Fog/smog/smoke/Sleet/hail (freezing rain or drizzle)': 8,
                  'Fog/smog/smoke/Snow': 9,
                  'Fog/smog/smoke/Unknown': 5,

                  'Not Reported': 0,

                  'Other': 0,
                  'Other/Blowing sand/snow': 1,
                  'Other/Clear': 3,
                  'Other/Cloudy': 4,
                  'Other/Fog/smog/smoke': 5,
                  'Other/Other': 0,
                  'Other/Rain': 6,
                  'Other/Severe crosswinds': 7,
                  'Other/Sleet/hail (freezing rain or drizzle)': 8,
                  'Other/Snow': 9,
                  'Other/Unknown': 0,

                  'Rain': 6,
                  'Rain/Blowing sand/snow': 1,
                  'Rain/Clear': 6,
                  'Rain/Cloudy': 6,
                  'Rain/Fog/smog/smoke': 6,
                  'Rain/Other': 6,
                  'Rain/Rain': 6,
                  'Rain/Reported but invalid': 6,
                  'Rain/Severe crosswinds': 10,
                  'Rain/Sleet/hail (freezing rain or drizzle)': 8,
                  'Rain/Snow': 9,
                  'Rain/Unknown': 6,

                  'Reported but invalid': 0,
                  'Reported but invalid/Reported but invalid': 0,

                  'Severe crosswinds': 7,
                  'Severe crosswinds/Blowing sand/snow': 2,
                  'Severe crosswinds/Clear': 7,
                  'Severe crosswinds/Cloudy': 7,
                  'Severe crosswinds/Other': 7,
                  'Severe crosswinds/Rain': 10,
                  'Severe crosswinds/Severe crosswinds': 7,
                  'Severe crosswinds/Snow': 11,
                  'Severe crosswinds/Unknown': 7,

                  'Sleet/hail (freezing rain or drizzle)': 8,
                  'Sleet/hail (freezing rain or drizzle)/Blowing sand/snow': 8,
                  'Sleet/hail (freezing rain or drizzle)/Clear': 8,
                  'Sleet/hail (freezing rain or drizzle)/Cloudy': 8,
                  'Sleet/hail (freezing rain or drizzle)/Fog/smog/smoke': 8,
                  'Sleet/hail (freezing rain or drizzle)/Other': 8,
                  'Sleet/hail (freezing rain or drizzle)/Rain': 8,
                  'Sleet/hail (freezing rain or drizzle)/Severe crosswinds': 12,
                  'Sleet/hail (freezing rain or drizzle)/Sleet/hail (freezing rain or drizzle)': 8,
                  'Sleet/hail (freezing rain or drizzle)/Snow': 8,
                  'Sleet/hail (freezing rain or drizzle)/Unknown': 8,

                  'Snow': 9,
                  'Snow/Blowing sand/snow': 9,
                  'Snow/Clear': 9,
                  'Snow/Cloudy': 9,
                  'Snow/Fog/smog/smoke': 9,
                  'Snow/Other': 9,
                  'Snow/Rain': 9,
                  'Snow/Reported but invalid': 9,
                  'Snow/Severe crosswinds': 11,
                  'Snow/Sleet/hail (freezing rain or drizzle)': 9,
                  'Snow/Snow': 9,
                  'Snow/Unknown': 9,

                  'Unknown': 0,
                  'Unknown/Blowing sand/snow': 1,
                  'Unknown/Clear': 3,
                  'Unknown/Cloudy': 4,
                  'Unknown/Other': 0,
                  'Unknown/Rain': 6,
                  'Unknown/Reported but invalid': 0,
                  'Unknown/Sleet/hail (freezing rain or drizzle)': 8,
                  'Unknown/Snow': 9,
                  'Unknown/Unknown': 0
                  }

    roadConditionMap = {'Dry': 1, 'Ice': 2,
                        'Not reported': 0, 'Other': 0,
                        'Reported but invalid': 0, 'Sand/mud/dirt/oil/gravel': 3,
                        'Slush': 4, 'Snow': 5,
                        'Unknown': 0, 'Water (standing - moving)': 6,
                        'Wet': 7
                        }

    # replacing the letters
    scaled_dataset = dataset.replace({
        'LightLevel': lightLevelMap,
        'Weather': weatherMap,
        'RoadCondition': roadConditionMap
    })

    return scaled_dataset


def balanceDataset(scaled_dataset):
    #scaled_dataset = scaled_dataset[:10000]
    # Take the continuous longitude/latitude
    coords = scaled_dataset[["Longitude", "Latitude"]].to_numpy()

    # Discretize them into, say, 10 bins each
    discretizer = KBinsDiscretizer(
        n_bins=10,          # number of bins per dimension
        encode="ordinal",   # returns integers 0..n_bins-1
        strategy="quantile" # each bin has roughly same number of samples
    )

    y_binned = discretizer.fit_transform(coords).astype(int)

    # y now becomes integer class labels for each coordinate dimension
    dataset_results = pd.DataFrame(
        y_binned,
        columns=["Longitude_bin", "Latitude_bin"]
    )

    # X = all other columns
    dataset_balanced = scaled_dataset.drop(columns=["Longitude", "Latitude"])
    return dataset_balanced, dataset_results


# =====================
#  PyTorch model & training
# =====================

class MultiOutputMLP(nn.Module):
    """
    Simple MLP with shared body and two separate heads:
    one for Longitude_bin, one for Latitude_bin.
    """
    def __init__(self, input_dim, hidden_sizes=(100,), n_classes=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)

        self.head_lon = nn.Linear(in_dim, n_classes)
        self.head_lat = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        h = self.shared(x)
        out_lon = self.head_lon(h)
        out_lat = self.head_lat(h)
        return out_lon, out_lat


def train_one_model(
    X_train, y_train,
    X_val, y_val,
    input_dim,
    hidden_sizes=(100,),
    n_classes=10,
    lr=1e-3,
    batch_size=128,
    epochs=50,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiOutputMLP(input_dim, hidden_sizes, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert numpy arrays to tensors
    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_lon_t = torch.from_numpy(y_train[:, 0].astype(np.int64)).to(device)
    y_train_lat_t = torch.from_numpy(y_train[:, 1].astype(np.int64)).to(device)

    train_ds = TensorDataset(X_train_t, y_train_lon_t, y_train_lat_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_lon = y_val[:, 0]
    y_val_lat = y_val[:, 1]

    for epoch in range(epochs):
        model.train()
        for xb, yb_lon, yb_lat in train_loader:
            optimizer.zero_grad()
            logits_lon, logits_lat = model(xb)
            loss_lon = criterion(logits_lon, yb_lon)
            loss_lat = criterion(logits_lat, yb_lat)
            loss = loss_lon + loss_lat
            loss.backward()
            optimizer.step()

    # Evaluate on validation set using MCC
    model.eval()
    with torch.no_grad():
        logits_lon_val, logits_lat_val = model(X_val_t)
        pred_lon = logits_lon_val.argmax(dim=1).cpu().numpy()
        pred_lat = logits_lat_val.argmax(dim=1).cpu().numpy()

    y_val_pred = np.column_stack([pred_lon, pred_lat])
    y_val_true = np.column_stack([y_val_lon, y_val_lat])
    val_mcc = mcc_multioutput(y_val_true, y_val_pred)

    return model, val_mcc


def evaluate_model(model, X_test, y_test, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        logits_lon, logits_lat = model(X_test_t)
        pred_lon = logits_lon.argmax(dim=1).cpu().numpy()
        pred_lat = logits_lat.argmax(dim=1).cpu().numpy()

    y_pred = np.column_stack([pred_lon, pred_lat])
    return mcc_multioutput(y_test, y_pred)


def trainAndTest(X_train, X_test, y_train, y_test):
    # Convert pandas DataFrames to numpy
    X_train_np = X_train.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.int64)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.int64)

    input_dim = X_train_np.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a small validation split from training data for model selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42
    )

    # Manual "param grid" for PyTorch model
    param_grid = [
        {"hidden_sizes": (50,),    "lr": 1e-3},
        {"hidden_sizes": (100,),   "lr": 1e-3},
        {"hidden_sizes": (50, 50), "lr": 5e-4},
    ]

    best_mcc = -np.inf
    best_params = None
    best_model = None

    print("Step 6 - Training (PyTorch MLP) with simple hyperparameter search:")
    for params in param_grid:
        print(f"  Trying params: {params}")
        model, val_mcc = train_one_model(
            X_tr, y_tr, X_val, y_val,
            input_dim=input_dim,
            hidden_sizes=params["hidden_sizes"],
            lr=params["lr"],
            device=device,
            batch_size=128,
            epochs=50
        )
        print(f"    Validation MCC: {val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_params = params
            best_model = model

    # Pretty-print training summary
    print(
        tabulate(
            [
                ["PyTorch Multi-layer Neural Network",
                 f"{best_params}",
                 best_mcc]
            ],
            headers=[
                "ML Trained Model",
                "Its Best Set of Parameter Values",
                "Its MCC-score on Validation Split (from Training Data)"
            ]
        )
    )

    # Evaluate on test data
    test_mcc = evaluate_model(best_model, X_test_np, y_test_np, device=device)

    print("Step 7 - Testing Data Summary:")
    print(
        tabulate(
            [
                ["PyTorch Multi-layer Neural Network",
                 f"{best_params}",
                 test_mcc]
            ],
            headers=[
                "ML Trained Model",
                "Its Best Set of Parameter Values",
                "Its MCC-score on Testing Data (20%)"
            ]
        )
    )

    return {"PyTorch_MLNN": test_mcc}


def main():
    filename = "oddYears(2005-2019).csv"
    original_dataset = getDataset(filename)
    scaled_dataset = scaleData(original_dataset)
    balanced_dataset, dataset_results = balanceDataset(scaled_dataset)

    # Step 6 - Splitting the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_dataset, dataset_results, test_size=0.2, random_state=42
    )

    test_mcc_scores = trainAndTest(X_train, X_test, y_train, y_test)

    print()
    print(f"The Best Overall Model is: {max(test_mcc_scores, key=test_mcc_scores.get)}")


if __name__ == "__main__":
    main()
