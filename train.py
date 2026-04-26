import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import preprocess

def train():
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data/spotify.csv")

    print("[INFO] Preprocessing...")
    X_scaled = preprocess(df)

    print("[INFO] Applying KMeans clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    print("[INFO] Clustering complete!")

    # ✅ SAVE MODEL
    joblib.dump(kmeans, "models/kmeans.pkl")

    # ✅ SAVE DATA
    df.to_csv("data/clustered_spotify.csv", index=False)

    # -------------------------------
    # 📊 PCA VISUALIZATION (NEW)
    # -------------------------------
    print("[INFO] Creating visualization...")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', s=10)

    plt.title("Spotify Song Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.savefig("data/cluster_plot.png")
    plt.show()

    print("[INFO] Visualization saved!")

if __name__ == "__main__":
    train()