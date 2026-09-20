import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SportyBet AI Slip Generator", layout="centered")
st.title("⚽ SportyBet AI Value Slip Generator")
st.write("Your automated daily value-betting and bankroll management assistant.")

@st.cache_resource
def load_and_train_model():
    if not os.path.exists('Matches.csv'):
        st.error("🚨 Error: 'Matches.csv' is missing from the repository root directory.")
        st.stop()
        
    matches = pd.read_csv('Matches.csv', low_memory=False)
    
    # Map columns based on actual dataset structure
    target_col = 'Full time result' if 'Full time result' in matches.columns else 'FTResult'
    home_odds_col = 'Bet365 home odds' if 'Bet365 home odds' in matches.columns else 'OddHome'
    home_goals_col = 'Full time home goals' if 'Full time home goals' in matches.columns else 'FTHome'
    away_goals_col = 'Full time away goals' if 'Full time away goals' in matches.columns else 'FTAway'
    
    # Feature engineering: compute goal difference
    matches['GoalDiff'] = matches[home_goals_col] - matches[away_goals_col]
    matches['HomeWin'] = (matches[target_col] == 'H').astype(int)
    
    # Use available numerical columns for features
    features = ['GoalDiff']
    if 'Home shots' in matches.columns:
        matches['ShotDiff'] = matches['Home shots'] - matches['Away shots']
        features.append('ShotDiff')
    
    model_df = matches.dropna(subset=[target_col, home_odds_col] + features).copy()
    
    X = model_df[features]
    y = model_df['HomeWin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, matches, home_odds_col, features

with st.spinner("Training AI Model on historical data..."):
    model, matches_df, home_odds_col, features = load_and_train_model()

st.success("AI Model Ready!")

st.sidebar.header("Settings & Bankroll")
bankroll = st.sidebar.number_input("Total Bankroll (NGN)", min_value=1000, value=50000, step=1000)

st.subheader("Upload Today's Fixtures")
uploaded_file = st.file_uploader("Upload your daily fixtures CSV file", type=["csv"])

if uploaded_file is not None:
    fixtures_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(fixtures_df)} incoming fixtures from your upload.")
else:
    st.info("No file uploaded yet. Using default sample batch from historical records for demonstration.")
    fixtures_df = matches_df.tail(30).dropna(subset=[home_odds_col])

if st.button("Generate Today's Betting Slips 🔥"):
    st.write("Scanning fixtures for positive Expected Value (+EV)...")
    
    slips = []
    for _, row in fixtures_df.iterrows():
        try:
            home = row['HomeTeam']
            away = row['AwayTeam']
            odds = row[home_odds_col]
            goal_diff = row.get('GoalDiff', 0)
            shot_diff = row.get('ShotDiff', 0)
        except KeyError:
            continue
        
        match_features = [[goal_diff] + ([shot_diff] if 'ShotDiff' in features else [])]
        
        ai_prob = model.predict_proba(match_features)[0][1]
        ev = (ai_prob * odds) - 1
        
        if ev > 0:
            b = odds - 1
            kelly = (b * ai_prob - (1 - ai_prob)) / b if b > 0 else 0
            stake = round(bankroll * max(0, kelly) * 0.5, 2)
            
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
