import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('data/oddYears(2005-2019).csv')

# Features and labels (you'll need to add a 'Risk' column or use clustering)
features = ['LightLevel', 'Weather', 'RoadCondition', 'Longitude', 'Latitude']

# For now, let's assume we're clustering high-risk zones
from sklearn.cluster import DBSCAN
coords = df[['Longitude', 'Latitude']].values
clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
df['RiskCluster'] = clustering.labels_

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(clustering, f)

print("Model trained and saved!")