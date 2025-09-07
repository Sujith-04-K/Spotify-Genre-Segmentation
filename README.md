<img width="909" height="804" alt="Spotify pic" src="https://github.com/user-attachments/assets/b8cb2288-bd9b-49a2-aa78-1cdaf1087872" />
# Spotify-Genre-Segmentation
ML project to classify Spotify songs by genre using audio features



# 🎵 Spotify Songs Genre Segmentation

## 📌 Overview
This project predicts the **genre of a song** using its **Spotify audio features**.  
It is a **multi-class classification problem** solved using **Machine Learning**.

---

## ⚙️ Features
- Data preprocessing & cleaning
- Exploratory Data Analysis (9 plots)
- ML Models: RandomForest & GradientBoosting
- Hyperparameter tuning (GridSearchCV)
- Saved best model: `spotify_genre_best_model.pkl`

---

## 📊 Visualizations
1. Genre Distribution  
2. Loudness Distribution  
3. Danceability Distribution  
4. Feature Correlation Heatmap  
5. Feature Histograms  
6. Confusion Matrix  
7. Feature Importances  
8. Tempo Distribution  
9. Valence vs Energy Scatterplot  

![Genre Distribution](images/genre_distribution.png)  
![Confusion Matrix](images/confusion_matrix.png)  
![Feature Importances](images/feature_importance.png)  

---

## 📈 Model Evaluation
- Accuracy: ~80–85%  
- Strong predictions for Pop, Rock, Hip-Hop  
- Overlap seen in Electronic vs Pop  
- Important features: Danceability, Energy, Tempo, Loudness

---

## 💾 Usage
```python
import pandas as pd, joblib

model = joblib.load("spotify_genre_best_model.pkl")
new_song = pd.DataFrame([{
    "danceability": 0.72, "energy": 0.81, "key": 5,
    "loudness": -6.5, "mode": 1, "speechiness": 0.06,
    "acousticness": 0.12, "instrumentalness": 0.0,
    "liveness": 0.09, "valence": 0.65, "tempo": 118.0,
    "duration_ms": 215000, "track_popularity": 70
}])
print("Predicted Genre:", model.predict(new_song)[0])







📌 1. Introduction

# 🎵 Spotify Songs Genre Segmentation

This project predicts the **genre of a song** based on its **Spotify audio features**.  
It is a **multi-class classification problem** solved using **Machine Learning**.

We perform:
- Exploratory Data Analysis (EDA) with 9 visualizations
- Model training with RandomForest & GradientBoosting
- Hyperparameter tuning with GridSearchCV
- Model evaluation (accuracy, classification report, confusion matrix)
- Save the best model for reuse




📌 2. Import Libraries & Load Dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load dataset
df = pd.read_csv("spotify dataset.csv")
df.columns = [c.strip() for c in df.columns]
print("✅ Data loaded:", df.shape)

# Preview
df.head()




📌 3. Data Overview

print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())




📌 4. Preprocessing


# Detect target column
target_col = "genre"  # update if different

# Encode target
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))
print("Target encoded. Classes:", list(le.classes_))

# Features
features = [
    'danceability','energy','key','loudness','mode','speechiness',
    'acousticness','instrumentalness','liveness','valence',
    'tempo','duration_ms','track_popularity'
]
X = df[features]
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)



📌 5. Exploratory Data Analysis (9 Plots)


# Apply theme
sns.set_theme(style="whitegrid")

# 1. Genre Distribution
plt.figure(figsize=(10,5))
sns.countplot(x=target_col, data=df, order=df[target_col].value_counts().index, palette="Set2", edgecolor="black")
plt.xticks(rotation=45)
plt.title("1. Genre Distribution")
plt.show()

# 2. Loudness Distribution
sns.histplot(df["loudness"], bins=30, kde=True, color="orange")
plt.title("2. Loudness Distribution (dB)")
plt.show()

# 3. Danceability Distribution
sns.histplot(df["danceability"], bins=30, kde=True, color="green")
plt.title("3. Danceability Distribution")
plt.show()

# 4. Feature Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[features].corr(), cmap="coolwarm", annot=False, square=True)
plt.title("4. Feature Correlation Heatmap")
plt.show()

# 5. Feature Histograms
df[features].hist(figsize=(15,12), bins=30, edgecolor="black")
plt.suptitle("5. Feature Distributions")
plt.show()

# (Training before confusion matrix needed, so we’ll do that below)



📌 6. Model Training


rf = RandomForestClassifier(random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(random_state=42)

# Cross-validation
rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
gb_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring="accuracy")

print("RandomForest CV Accuracy:", rf_scores.mean())
print("GradientBoosting CV Accuracy:", gb_scores.mean())

# GridSearch for RandomForest
param_grid = {"n_estimators":[100,200], "max_depth":[None,10,20]}
grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)

print("✅ Best Params:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




📌 7. Model Evaluation Visuals


# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("6. Confusion Matrix - Genre Prediction")
plt.show()

# 7. Feature Importances
fi = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=fi.values[:10], y=fi.index[:10], palette="viridis")
plt.title("7. Top 10 Important Features")
plt.show()

# 8. Tempo Distribution
sns.histplot(df["tempo"], bins=30, kde=True, color="purple")
plt.title("8. Tempo Distribution (BPM)")
plt.show()

# 9. Valence vs Energy Scatterplot
sns.scatterplot(x="valence", y="energy", hue=target_col, data=df, palette="Set1", alpha=0.7)
plt.title("9. Valence vs Energy (by Genre)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


📌 8. Test on New Song

example_song = pd.DataFrame([{
    "danceability": 0.72, "energy": 0.81, "key": 5,
    "loudness": -6.5, "mode": 1, "speechiness": 0.06,
    "acousticness": 0.12, "instrumentalness": 0.0,
    "liveness": 0.09, "valence": 0.65, "tempo": 118.0,
    "duration_ms": 215000, "track_popularity": 70
}])

pred = best_rf.predict(example_song)[0]
print("🎵 Predicted Genre:", le.inverse_transform([pred])[0])



📌 9. Conclusion

# ✅ Conclusion
- The model achieves ~80–85% accuracy on test data.
- Most influential features: **Danceability, Energy, Tempo, Loudness**.
- Strong predictions for Pop, Rock, Hip-Hop.
- Can be used for **playlist classification, recommendation systems, or arti



Project Structure 



spotify-genre-segmentation
│── spotify_genre_segmentation.ipynb   # Main notebook (code + plots)
│── spotify_genre_best_model.pkl       # Saved trained model
│── README.md                          # Project documentation
│── images/                            # Saved plots for README
│    ├── genre_distribution.png
│    ├── loudness_distribution.png
│    ├── danceability_distribution.png
│    ├── feature_heatmap.png
│    ├── feature_histograms.png
│    ├── confusion_matrix.png
│    ├── feature_importance.png
│    ├── tempo_distribution.png
│    ├── valence_vs_energy.png
