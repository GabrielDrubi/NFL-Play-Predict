import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

gbc = data["model"]
le_posteam = data["le_posteam"]
le_posteam_type = data["le_posteam_type"]

def show_predict_page():
    
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html= True)
    
    form = st.form("name")
    
    st.title("NFL Play Prediction")

    st.write("""### We need some information to predict the play""")

    posteam = (
        "NE",
        "NO",
        "HOU",
        "PHI",
        "DET",
        "DEN",
        "NYJ",
        "ATL",
        "IND",
        "GB",
        "CAR",
        "CIN",
        "PIT",
        "KC",
        "BUF",
        "NYG",
        "DAL",
        "MIN",
        "OAK",
        "WAS",
        "SEA",
        "TB",
        "SF",
        "ARI",
        "MIA",
        "CHI",
        "CLE",
        "TEN",
        "SD",
        "JAC",
        "STL",
        "JAX",
        "LA",
        "LAC"
    )

    
    qtr = (
        "1",
        "2",
        "3",
        "4",
        "5"
    )
    
    down = (
        "1",
        "2",
        "3",
        "4",
    )
    yardstogo = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
    )

    posteam = st.selectbox("Possesion Team", posteam)
    posteam_type = st.radio(
    "Possesion Team Status: ",
    ('home', 'away'))
    qtr = st.selectbox("Quarter (5 =  overtime)", qtr)
    down = st.selectbox("Down", down)
    yardstogo = st.selectbox("Yards To First Down", yardstogo)
    PassingMean = st.number_input('Passing Mean')
    RushingMean = st.number_input('Rushing Mean')
    
    posteam_timeouts_remaining = st.number_input("Possesion Team Timeout Remaining", min_value=0, max_value=4, value=0)
    game_seconds_remaining = st.slider("Game Seconds Remaining", 0, 3600, 500)
    half_seconds_remaining = st.slider("Half Seconds Remaining", 0, 900, 30)
    score_differential = st.slider("Score Differential (According on the Possesion Team)", -100, 100, 0)
    yardline_100 = st.slider("Yard Line", 1, 99, 15)
    
    ok = st.button("Predict Play")
    if ok:
        X = np.array([[qtr, yardline_100, yardstogo, half_seconds_remaining, down, score_differential, posteam, posteam_type, game_seconds_remaining, posteam_timeouts_remaining, PassingMean, RushingMean]])
        X[:, 6] = le_posteam.transform(X[:,6])
        X[:, 7] = le_posteam_type.transform(X[:,7])
        X = X.astype(float)

        play = gbc.predict(X)
        st.subheader(f"It will be a {play[0]} play")
    
    

