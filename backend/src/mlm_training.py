# this should let gpu acceleration run
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on the earth.
    Returns distance in miles.
    """
    R = 3958.8  # Radius of earth in miles

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def custom_score_one(x_vals, y_pred, binned_coords, tolerance_miles=0.5):
    """
    Score = distance (miles) from predicted coords to the nearest coord
    in the training bin for these x_vals.
    If nearest distance <= tolerance, return 0.0.
    If bin is empty, return +inf (or you can choose another fallback).
    """
    key = (int(x_vals[0]), int(x_vals[1]), int(x_vals[2]))
    candidates = binned_coords.get(key, None)

    pred_lon, pred_lat = float(y_pred[0]), float(y_pred[1])

    if candidates is None:
        return float("inf")

    # candidates may be a torch tensor on GPU (K,2) [lon,lat] or a python list
    if torch.is_tensor(candidates):
        cand_np = candidates.detach().cpu().numpy()
    else:
        cand_np = np.asarray(candidates, dtype=np.float32)

    if cand_np.size == 0:
        return float("inf")

    best = float("inf")
    for lon, lat in cand_np:
        d = haversine_distance(lat, lon, pred_lat, pred_lon)
        if d < best:
            best = d

    return 0.0 if best <= tolerance_miles else best

def custom_score_batch(X_np, y_pred_np, binned_coords, tolerance_miles=0.5):
    # X_np: (N,3), y_pred_np: (N,2)
    scores = []
    for x, yp in zip(X_np, y_pred_np):
        cs = custom_score_one(x, yp, binned_coords, tolerance_miles=tolerance_miles)
        if (cs != float("inf")):
            scores.append(cs)

    return float(np.mean(scores))

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
        'Not reported': 0,
        'Other': 0,
        'Reported but invalid': 0,
        'Unknown': 0,
        'Dawn': 1,
        'Dusk': 1,
        'Daylight': 2,
        'Dark - lighted roadway': 3,
        'Dark - roadway not lighted': 3,
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
                  "Fog/smog/smoke/Blowing sand/snow": 1,
                  "Fog/smog/smoke/Severe crosswinds": 13,

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
                  "Reported but invalid/Clear": 3,
                  "Reported but invalid/Cloudy": 4,
                  "Reported but invalid/Rain": 6,

                  'Severe crosswinds': 7,
                  'Severe crosswinds/Blowing sand/snow': 2,
                  'Severe crosswinds/Clear': 7,
                  'Severe crosswinds/Cloudy': 7,
                  'Severe crosswinds/Other': 7,
                  'Severe crosswinds/Rain': 10,
                  'Severe crosswinds/Severe crosswinds': 7,
                  'Severe crosswinds/Snow': 11,
                  'Severe crosswinds/Unknown': 7,
                  "Severe crosswinds/Fog/smog/smoke": 13,
                  "Severe crosswinds/Sleet/hail (freezing rain or drizzle)": 12,

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
                  "Sleet/hail (freezing rain or drizzle)/Reported but invalid": 8,

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
                  'Unknown/Unknown': 0,
                  "Unknown/Severe crosswinds": 7
                  }

    roadConditionMap = {'Dry': 1,
                        'Ice': 2,
                        'Not reported': 0,
                        'Other': 0,
                        'Reported but invalid': 0,
                        'Sand/mud/dirt/oil/gravel': 3,
                        'Slush': 4,
                        'Snow': 5,
                        'Unknown': 0,
                        'Water (standing - moving)': 6,
                        'Wet': 7
                        }

    # replacing the letters
    scaled_dataset = dataset.replace({
        'LightLevel': lightLevelMap,
        'Weather': weatherMap,
        'RoadCondition': roadConditionMap
    }).infer_objects(copy=False)

    return scaled_dataset

def balanceDataset(scaled_dataset):
    # y = continuous longitude / latitude
    y = scaled_dataset[["Longitude", "Latitude"]]

    # X = remaining features
    X = scaled_dataset.drop(columns=["Longitude", "Latitude"])

    return X, y

# =====================
#  PyTorch model & training
# =====================

# ---------------------
# Distance (Torch) - differentiable
# ---------------------
def haversine_torch(lat1, lon1, lat2, lon2):
    """
    lat1, lon1: scalars or tensors
    lat2, lon2: tensors broadcastable to lat1/lon1
    returns: distance in miles
    """
    R = 3958.8
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c

# ---------------------
# Build bin -> candidate coords (ONCE, from training split)
# ---------------------
def build_binned_coords_torch(X_np, y_np, device):
    """
    X_np: (N, 3) float32 or similar with [LightLevel, Weather, RoadCondition]
    y_np: (N, 2) float32 with [Longitude, Latitude]
    returns: dict[(ll,w,rc)] -> tensor (K,2) with [lon, lat] on GPU
    """
    tmp = defaultdict(list)
    for x, y in zip(X_np, y_np):
        key = (int(x[0]), int(x[1]), int(x[2]))
        tmp[key].append([float(y[0]), float(y[1])])  # [lon, lat]

    out = {}
    for k, coords in tmp.items():
        out[k] = torch.tensor(coords, dtype=torch.float32, device=device)
    return out

# ---------------------
# Differentiable custom loss: soft-min haversine to candidates in the bin
# ---------------------
class SoftBinHaversineLoss(nn.Module):
    def __init__(self, binned_coords, tau=0.25):
        """
        tau controls how close soft-min is to true min (smaller = closer, but can be harder to optimize).
        """
        super().__init__()
        self.bins = binned_coords
        self.tau = float(tau)

    def forward(self, preds_lonlat, x_vals):
        """
        preds_lonlat: (B, 2) -> [lon, lat]
        x_vals:       (B, 3) -> [LightLevel, Weather, RoadCondition]
        returns: scalar loss (mean soft-min distance in miles)
        """
        losses = []

        # NOTE: bins are variable-length lists; easiest is loop per sample (still GPU for math)
        for i in range(preds_lonlat.size(0)):
            key = (
                int(x_vals[i, 0].item()),
                int(x_vals[i, 1].item()),
                int(x_vals[i, 2].item())
            )
            candidates = self.bins.get(key, None)
            if candidates is None or candidates.numel() == 0:
                continue  # if a bin is empty, just skip this sample

            lon_p = preds_lonlat[i, 0]
            lat_p = preds_lonlat[i, 1]

            # candidates: (K,2) = [lon, lat]
            dists = haversine_torch(
                lat_p, lon_p,
                candidates[:, 1], candidates[:, 0]
            )  # (K,)

            # soft-min: -tau * logsumexp(-d/tau)
            tau = self.tau
            soft_min = -tau * torch.logsumexp(-dists / tau, dim=0)
            losses.append(soft_min)

        if not losses:
            # if everything got skipped (rare), return 0 with grad
            return (preds_lonlat.sum() * 0.0)

        return torch.stack(losses).mean()

# =====================
#  Model
# =====================
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(100,)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)  # [lon, lat]

    def forward(self, x):
        return self.head(self.shared(x))

# =====================
#  Training / Eval using ONLY custom loss
# =====================
def train_one_model(
    X_train, y_train,
    X_val, y_val,
    input_dim,
    binned_coords,
    hidden_sizes=(100,),
    lr=1e-3,
    batch_size=128,
    epochs=50,
    tau=0.25,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiOutputMLP(input_dim, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SoftBinHaversineLoss(binned_coords, tau=tau)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)                    # (B,2) lon/lat
            loss = criterion(preds, xb[:, :3])   # use x bins (first 3 cols)
            loss.backward()
            optimizer.step()

    # Validation score = same custom loss (lower is better)
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val_t)
        val_loss = criterion(preds_val, X_val_t[:, :3]).item()

    return model, val_loss

def evaluate_model(model, X_test, binned_coords, tau=0.25, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    X_test_t = torch.from_numpy(X_test).to(device)

    with torch.no_grad():
        preds = model(X_test_t)  # (N,2) tensor on device

    preds_np = preds.detach().cpu().numpy()
    return custom_score_batch(X_test, preds_np, binned_coords)

def trainAndTest(X_train, X_test, y_train, y_test, tau=0.25):
    X_train_np = X_train.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    X_test_np  = X_test.to_numpy(dtype=np.float32)
    y_test_np  = y_test.to_numpy(dtype=np.float32)

    input_dim = X_train_np.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42
    )

    # Build bins from TRAINING split (X_tr/y_tr), on GPU
    binned_coords = build_binned_coords_torch(X_tr, y_tr, device=device)

    param_grid = [
        {"hidden_sizes": (50,),    "lr": 1e-3},
        {"hidden_sizes": (100,),   "lr": 1e-3},
        {"hidden_sizes": (50, 50), "lr": 5e-4},
    ]

    best_score = float("inf")
    best_params = None
    best_model = None

    print("Step 6 - Training (PyTorch MLP) optimizing ONLY custom loss (soft-min haversine):")
    for params in param_grid:
        print(f"  Trying params: {params}")
        model, val_score = train_one_model(
            X_tr, y_tr, X_val, y_val,
            input_dim=input_dim,
            binned_coords=binned_coords,
            hidden_sizes=params["hidden_sizes"],
            lr=params["lr"],
            device=device,
            batch_size=128,
            epochs=50,
            tau=tau
        )
        print(f"    Validation custom loss (mean miles, soft-min): {val_score:.4f}")

        if val_score < best_score:
            best_score = val_score
            best_params = params
            best_model = model

    print(tabulate(
        [["PyTorch MLP (custom loss only)", f"{best_params}", best_score]],
        headers=["Model", "Best Params", f"Val loss (mean miles; tau={tau})"]
    ))

    test_score = evaluate_model(best_model, X_test_np,
                                binned_coords=binned_coords, tau=tau, device=device)

    print("Step 7 - Testing Data Summary:")
    print(tabulate(
        [["PyTorch MLP (custom loss only)", f"{best_params}", test_score]],
        headers=["Model", "Best Params", f"Test loss (mean miles; tau={tau})"]
    ))

    #torch.save(best_model, "model_save_10000")


    return {"PyTorch_custom_loss_only": test_score}

def main():
    pd.set_option("future.no_silent_downcasting", True)
    filename = "allYears.csv"
    original_dataset = getDataset(filename)
    scaled_dataset = scaleData(original_dataset)
    balanced_dataset, dataset_results = balanceDataset(scaled_dataset[:10])

    # Step 6 - Splitting the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(balanced_dataset, dataset_results, test_size=0.2, random_state=42)
    trainAndTest(X_train, X_test, y_train, y_test)
    print(X_test.to_numpy(dtype=np.float32))

if __name__ == "__main__":
    main()
