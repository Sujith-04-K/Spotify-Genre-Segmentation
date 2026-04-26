import streamlit as st
import pandas as pd

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Spotify AI", page_icon="🎧", layout="wide")

# -------------------------------
# 🎨 SPOTIFY STYLE UI
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #121212;
    color: #FFFFFF;
    font-family: 'Segoe UI';
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #1DB954;
}

/* Cards */
.card {
    background-color: #181818;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #282828;
}

/* Buttons */
.stButton>button {
    background-color: #1DB954;
    color: black;
    font-size: 18px;
    border-radius: 25px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    border: none;
}

/* Hover */
.stButton>button:hover {
    background-color: #1ed760;
}

/* Headers */
h2 {
    color: #1DB954;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🎧 TITLE
# -------------------------------
st.markdown('<div class="title">🎧 Spotify AI Recommender</div>', unsafe_allow_html=True)
st.markdown("### Discover similar music using AI clustering")

# -------------------------------
# 📂 LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/clustered_spotify.csv")

df = load_data()

# -------------------------------
# 🎵 SONG SELECT
# -------------------------------
st.markdown("## 🎵 Select a Song")

song = st.selectbox("Choose a track", df['track_name'].values)

selected = df[df['track_name'] == song].iloc[0]
cluster = selected['cluster']

# -------------------------------
# 📊 SONG DETAILS
# -------------------------------
st.markdown("## 📊 Song Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"🎤 Artist: {selected['track_artist']}")
    st.write(f"💿 Album: {selected['track_album_name']}")
    st.write(f"🔥 Popularity: {selected['track_popularity']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"⚡ Energy: {round(selected['energy'],2)}")
    st.write(f"💃 Danceability: {round(selected['danceability'],2)}")
    st.write(f"🎼 Tempo: {round(selected['tempo'],2)}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 🎯 CLUSTER INFO
# -------------------------------
st.markdown("## 🎯 Cluster Type")

cluster_names = {
    0: "🎧 Chill / Acoustic",
    1: "🎉 Party / High Energy",
    2: "💃 Dance Music",
    3: "😌 Relaxed / Calm",
    4: "🔥 Mixed Style"
}

st.success(f"{cluster_names.get(cluster, 'Unknown Cluster')}")

# -------------------------------
# 🔁 SMART RECOMMENDATIONS
# -------------------------------
st.markdown("## 🔁 Recommended Songs")

similar = df[df['cluster'] == cluster]

# Avoid same song
similar = similar[similar['track_name'] != song]

# Sort by popularity (better than random)
similar = similar.sort_values(by='track_popularity', ascending=False).head(5)

for i, row in similar.iterrows():
    st.write(f"🎵 {row['track_name']} — {row['track_artist']}")

# -------------------------------
# 📊 CLUSTER INSIGHTS
# -------------------------------
st.markdown("## 📊 Cluster Insights")

cluster_stats = df.groupby('cluster')[[
    'energy','danceability','valence'
]].mean().round(2)

st.dataframe(cluster_stats)

# -------------------------------
# 📈 VISUALIZATION
# -------------------------------
st.markdown("## 📈 Cluster Visualization")

st.image("data/cluster_plot.png", caption="PCA Cluster Distribution")