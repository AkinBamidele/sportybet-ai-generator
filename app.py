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
    
    # Clean up column names in case of whitespace
    matches.columns = matches.columns.str.strip()
    
    # Flexible column mapping for target and odds
    target_col = 'FTResult' if 'FTResult' in matches.columns else ('Full time result' if 'Full time result' in matches.columns else None)
    home_odds_col = 'OddHome' if 'OddHome' in matches.columns else ('Bet365 home odds' if 'Bet365 home odds' in matches.columns else None)
    
    if not target_col or not home_odds_col:
        st.error(f"🚨 Error: Could not find target result or home odds columns. Available: {list(matches.columns)}")
        st.stop()

    # Define desired features and check if they exist
    potential_features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
    available_features = [f for f in potential_features if f in matches.columns]
    
    if not available_features:
        home_goals = 'FTHome' if 'FTHome' in matches.columns else 'Full time home goals'
        away_goals = 'FTAway' if 'FTAway' in matches.columns else 'Full time away goals'
        if home_goals in matches.columns and away_goals in matches.columns:
            matches['GoalDiff'] = matches[home_goals] - matches[away_goals]
            available_features = ['GoalDiff']
        else:
            st.error("🚨 Error: Insufficient columns for model training in Matches.csv.")
            st.stop()

    # Drop NA only on columns that actually exist
    subset_cols = [target_col, home_odds_col] + available_features
    model_df = matches.dropna(subset=subset_cols).copy()
    
    if 'HomeElo' in available_features and 'AwayElo' in available_features:
        model_df['EloDiff'] = model_df['HomeElo'] - model_df['AwayElo']
        if 'EloDiff' not in available_features:
            available_features.insert(0, 'EloDiff')
            
    model_df['HomeWin'] = (model_df[target_col].astype(str).str.upper() == 'H').astype(int)

    X = model_df[available_features]
    y = model_df['HomeWin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, matches, home_odds_col, available_features

with st.spinner("Training AI Model on historical data..."):
    model, matches_df, home_odds_col, features = load_and_train_model()

st.success("AI Model Ready!")

st.sidebar.header("Settings & Bankroll")
bankroll = st.sidebar.number_input("Total Bankroll (NGN)", min_value=1000, value=50000, step=1000)

st.subheader("Upload Today's Fixtures")
uploaded_file = st.file_uploader("Upload your daily fixtures CSV file", type=["csv"])

if uploaded_file is not None:
    fixtures_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    fixtures_df.columns = fixtures_df.columns.str.strip()
    st.write(f"Loaded {len(fixtures_df)} incoming fixtures from your upload.")
else:
    st.info("No file uploaded yet. Using default sample batch from historical records for demonstration.")
    fixtures_df = matches_df.tail(30).dropna(subset=[home_odds_col])

if st.button("Generate Today's Betting Slips 🔥"):
    st.write("Scanning fixtures for positive Expected Value (+EV)...")
    
    slips = []
    for _, row in fixtures_df.iterrows():
        try:
            home = row.get('HomeTeam', 'Home')
            away = row.get('AwayTeam', 'Away')
            odds = row[home_odds_col]
            
            match_feat_vals = []
            for f in features:
                if f == 'EloDiff':
                    match_feat_vals.append(row.get('HomeElo', 0) - row.get('AwayElo', 0))
                else:
                    match_feat_vals.append(row.get(f, 0))
        except Exception:
            continue
        
        match_features = [match_feat_vals]
        
        try:
            ai_prob = model.predict_proba(match_features)[0][1]
        except Exception:
            continue
            
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
