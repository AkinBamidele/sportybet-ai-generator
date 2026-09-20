import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration & Header
st.set_page_config(page_title="SportyBet AI Slip Generator", layout="centered")
st.title("⚽ SportyBet AI Value Slip Generator")
st.write("Your automated daily value-betting and bankroll management assistant.")

# Cache model training for high performance and speed
@st.cache_resource
def load_and_train_model():
    if not os.path.exists('Matches.csv'):
        st.error("🚨 Error: 'Matches.csv' is missing from the repository root directory.")
        st.stop()
        
    matches = pd.read_csv('Matches.csv', low_memory=False)
    model_df = matches.dropna(subset=['FTResult', 'HomeElo', 'AwayElo', 'Form5Home', 'Form5Away', 'OddHome']).copy()
    model_df['EloDiff'] = model_df['HomeElo'] - model_df['AwayElo']
    model_df['HomeWin'] = (model_df['FTResult'] == 'H').astype(int)

    features = ['EloDiff', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
    X = model_df[features]
    y = model_df['HomeWin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, matches

with st.spinner("Training AI Model on historical data..."):
    model, matches_df = load_and_train_model()

st.success("AI Model Ready!")

# Sidebar Settings
st.sidebar.header("Settings & Bankroll")
bankroll = st.sidebar.number_input("Total Bankroll (NGN)", min_value=1000, value=50000, step=1000)

# File Uploader for Mobile / Daily Fixtures
st.subheader("Upload Today's Fixtures")
uploaded_file = st.file_uploader("Upload your daily fixtures CSV file", type=["csv"])

if uploaded_file is not None:
    fixtures_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(fixtures_df)} incoming fixtures from your upload.")
else:
    st.info("No file uploaded yet. Using default sample batch from historical records for demonstration.")
    fixtures_df = matches_df.tail(30).dropna(subset=['HomeElo', 'AwayElo', 'OddHome'])

# Action Trigger
if st.button("Generate Today's Betting Slips 🔥"):
    st.write("Scanning fixtures for positive Expected Value (+EV)...")
    
    slips = []
    for _, row in fixtures_df.iterrows():
        try:
            home = row['HomeTeam']
            away = row['AwayTeam']
            h_elo = row['HomeElo']
            a_elo = row['AwayElo']
            f3_h = row['Form3Home']
            f5_h = row['Form5Home']
            f3_a = row['Form3Away']
            f5_a = row['Form5Away']
            odds = row['OddHome']
        except KeyError:
            continue
        
        elo_diff = h_elo - a_elo
        match_features = [[elo_diff, h_elo, a_elo, f3_h, f5_h, f3_a, f5_a]]
        
        ai_prob = model.predict_proba(match_features)[0][1]
        ev = (ai_prob * odds) - 1
        
        if ev > 0:
            b = odds - 1
            kelly = (b * ai_prob - (1 - ai_prob)) / b
            stake = round(bankroll * kelly * 0.5, 2)
            
            slips.append({
                'Match': f"{home} vs {away}",
                'AI Win Prob': f"{ai_prob * 100:.1f}%",
                'Odds': odds,
                'EV': f"{ev * 100:.2f}%",
                'Stake (NGN)': f"₦{stake:,.2f}"
            })
            
    slips_df = pd.DataFrame(slips)
    
    if not slips_df.empty:
        st.balloons()
        st.subheader(f"Found {len(slips_df)} Value Bets Today:")
        st.dataframe(slips_df, use_container_width=True)
        
        csv_data = slips_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Betting Slips CSV",
            data=csv_data,
            file_name="ready_to_play_slips.csv",
            mime="text/csv",
        )
    else:
        st.warning("No positive EV bets found in today's batch. Safe to skip!")
