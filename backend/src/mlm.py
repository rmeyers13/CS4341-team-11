import numpy as np
import torch
from mlm_training import MultiOutputMLP

device = None
if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("model_save_10000", weights_only=False)
model.to(device)
model.eval()


# This takes an array of 3 values (in order 'LightLevel', 'Weather', 'RoadCondition') [0-3, 0-13, 0-7].to_numpy(dtype=np.float32) Example:{np.asarray([[0, 2, 3]], dtype=np.float32)} the maps for what those ints mean are in scaleData in mlm_training.py
#It returns two vars: long, lat 
def getCoords(testArray):
    X_test_t = torch.from_numpy(testArray).to(device)

    with torch.no_grad():
        preds = model(X_test_t)

    long, lat = preds[0].tolist()
    return long, lat