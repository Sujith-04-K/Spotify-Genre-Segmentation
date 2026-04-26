import pandas as pd
from sklearn.preprocessing import StandardScaler

# 🔥 Features used for clustering
FEATURES = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

def preprocess(df):
    print("[INFO] Selecting features...")

    # Select only required features
    X = df[FEATURES]

    print("[INFO] Handling missing values...")
    # Fill missing values with mean
    X = X.fillna(X.mean())

    print("[INFO] Scaling data...")
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Preprocessing complete!")

    return X_scaled