import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv('data/accidents.csv')

# Convert categorical data to numbers
df['LightLevel'] = df['LightLevel'].astype('category').cat.codes
df['Weather'] = df['Weather'].astype('category').cat.codes
df['RoadCondition'] = df['RoadCondition'].astype('category').cat.codes

# Save cleaned data
df.to_csv('data/accidents_clean.csv', index=False)