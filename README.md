# 🎧 Spotify Music Intelligence System
### AI-Based Song Clustering & Recommendation Engine

---

## 🔬 Overview

This project is an end-to-end Machine Learning system that analyzes Spotify song data to:

- 🎵 Segment songs into clusters using **unsupervised learning (KMeans)**
- 🔁 Recommend similar songs using **distance-based similarity**
- 📊 Provide insights into music patterns
- 🎨 Deliver an interactive **Spotify-style dashboard**

Unlike basic clustering projects, this system combines **Machine Learning + Visualization + Recommendation + UI**, making it a complete data-driven application.

---

## 🚀 Key Features

- 🎧 Spotify-inspired premium UI (Streamlit)  
- 🤖 KMeans clustering (unsupervised learning)  
- 🧠 Distance-based recommendation system (Euclidean similarity)  
- 🔍 Search songs by name or artist  
- 📊 Cluster insights and statistics  
- 📈 PCA-based cluster visualization  
- 💡 Explainable cluster behavior  

---

## 🧠 Machine Learning Pipeline

1. Data Loading  
2. Feature Selection (audio features)  
3. Data Preprocessing (handling missing values, scaling)  
4. Clustering using KMeans  
5. Dimensionality Reduction using PCA  
6. Recommendation using similarity (Euclidean distance)  

---

## 📂 Project Structure
spotify_project/
│── data/
│ ├── spotify.csv
│ ├── clustered_spotify.csv
│ └── cluster_plot.png
│
│── src/
│ ├── train.py
│ ├── utils.py
│ └── visuals.py
│
│── models/
│ └── kmeans.pkl
│
│── assets/
│ ├── cluster_plot.png
│ ├── energy_dist.png
│ ├── cluster_stats.png
│
│── app.py
│── requirements.txt
│── README.md



---

## 📊 Visual Insights

### 🎯 Cluster Visualization
![Clusters](assets/cluster_plot.png)

### ⚡ Feature Distribution
![Energy](assets/energy_dist.png)

### 📈 Cluster Comparison
![Cluster Stats](assets/cluster_stats.png)

---

## 🎧 Application Preview

### 🎵 Song Selection & Dashboard
![Dashboard](assets/image1.png)

### 🔁 Smart Recommendations
![Recommendations](assets/image2.png)

### 📊 Insights View
![Insights](assets/image3.png)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/Sujith-04-K/Spotify-Genre-Segmentation.git
cd Spotify-Genre-Segmentation



---

### 2️⃣ Install Dependencies
pip install -r requirements.txt



---

### 3️⃣ Train the Model
python src/train.py



---

### 4️⃣ Generate Visualizations
python src/visuals.py



---

### 5️⃣ Run the Application
streamlit run app.py



---

## 📥 Input Features

The model uses audio features such as:

- Danceability  
- Energy  
- Loudness  
- Speechiness  
- Acousticness  
- Instrumentalness  
- Liveness  
- Valence  
- Tempo  

---

## 🔁 Recommendation System

This system uses:

👉 **Euclidean Distance** to measure similarity between songs  

This ensures:
- More accurate recommendations  
- Better grouping of similar tracks  

---

## 📈 Output

The system provides:

- 🎯 Cluster assignment  
- 🔁 Similar song recommendations  
- 📊 Cluster insights (energy, danceability, mood)  
- 📈 Visual representation of clusters  

---

## 🎯 Key Highlight

> This project goes beyond traditional clustering by integrating a **distance-based recommendation system**, making it more practical and closer to real-world music platforms.

---

## ⚠️ Notes

- Model is unsupervised (no labeled data required)  
- Dataset must be placed inside `data/` folder  
- Run training script before launching UI  
- Visualizations are stored in `assets/` folder  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## 🚀 Future Improvements

- Cosine similarity for better recommendations  
- Integration with Spotify API  
- Real-time music recommendation  
- Deep learning models  

---

## 👨‍💻 Author

**Sujith K**  
🔗 GitHub: https://github.com/Sujith-04-K  

---

## ⭐ Support

If you found this project useful, please ⭐ the repository!
