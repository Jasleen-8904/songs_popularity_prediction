import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Spotify Song Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# -----------------------------
# Custom CSS for Better UI
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    background-color: #1DB954;
    color: white;
    border: none;
}

.stButton>button:hover {
    background-color: #1ed760;
    color: white;
}

.big-font {
    font-size: 32px !important;
    font-weight: 700;
    color: #1DB954;
}

.small-text {
    font-size: 18px;
    color: #CCCCCC;
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎵 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Predict Popularity", "About Project"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":

    st.markdown('<p class="big-font">Spotify Song Popularity Predictor</p>', unsafe_allow_html=True)

    st.markdown("""
    <p class="small-text">
    Predict whether a song has the potential to become popular using Machine Learning.
    This project uses audio features like BPM, Danceability, Energy, and more.
    </p>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", "50+ Songs")

    with col2:
        st.metric("Features Used", "9")

    with col3:
        st.metric("ML Model", "Regression")

    st.write("")
    st.subheader("📌 Features Used")

    st.write("""
    - Beats Per Minute (BPM)
    - Energy
    - Danceability
    - Loudness
    - Liveness
    - Valence
    - Length
    - Acousticness
    - Speechiness
    """)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Predict Popularity":

    st.title("Predict Song Popularity")

    st.write("Enter the song features below:")

    col1, col2 = st.columns(2)

    with col1:
        bpm = st.number_input("Beats Per Minute", 50, 250, 120)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        loudness = st.number_input("Loudness (dB)", -60, 10, -5)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.5)

    with col2:
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        length = st.number_input("Length (seconds)", 100, 500, 200)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)

    st.write("")

    if st.button("Predict Popularity 🚀"):

        input_data = pd.DataFrame({
            "Beats.Per.Minute": [bpm],
            "Energy": [energy],
            "Danceability": [danceability],
            "Loudness..dB..": [loudness],
            "Liveness": [liveness],
            "Valence.": [valence],
            "Length.": [length],
            "Acousticness..": [acousticness],
            "Speechiness.": [speechiness]
        })

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        popularity_score = round(float(prediction[0]), 2)

        st.success(f"🎵 Predicted Popularity Score: {popularity_score}/100")

        if popularity_score >= 80:
            st.balloons()
            st.success("🔥 High Hit Potential! This song can become very popular.")

        elif popularity_score >= 70:
            st.info("⭐ Good Potential! This song has decent popularity chances.")

        else:
            st.warning("🎯 Average Potential. The song may need improvement for higher popularity.")

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "About Project":

    st.title("About This Project")

    st.write("""
    ### Project Name:
    Spotify Song Popularity Predictor

    ### Objective:
    To predict the popularity of a song using Machine Learning based on its audio features.

    ### Technologies Used:
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Seaborn
    - Pickle

    ### Machine Learning Used:
    - Data Preprocessing
    - Feature Scaling
    - Regression Model

    ### Business Value:
    Helps music producers and artists understand which songs are more likely to perform well.

    ### Developed By:
    Jasleen Kaur
    BTech CSE | Data Science
    """)

# -----------------------------
# Footer
# -----------------------------
st.write("")
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")