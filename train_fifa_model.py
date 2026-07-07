import pandas as pd
import numpy as np
import pickle
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

def get_k_factor(tournament):
    t = str(tournament).lower()
    if "fifa world cup" in t and "qualifying" not in t:
        return 60
    elif "cup" in t or "copa" in t or "euro" in t or "championship" in t:
        if "qualifying" in t or "qualification" in t:
            return 30
        return 40
    elif "friendly" in t:
        return 20
    else:
        return 30

def train_model():
    # Try to use international_matches.csv first (more complete), fall back to updated.csv
    primary = "international_matches.csv"
    fallback = "updated.csv"

    if os.path.exists(primary):
        print(f"Loading {primary}...")
        df_hist = pd.read_csv(primary)
    else:
        print(f"{primary} not found, loading {fallback}...")
        df_hist = pd.read_csv(fallback)

    df_hist = df_hist.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)
    print(f"Total matches loaded: {len(df_hist)}")

    # Clean team names
    name_mapping = {
        "Korea Republic": "South Korea",
        "Czech Republic": "Czechia",
        "Turkey": "Türkiye",
        "Ivory Coast": "Côte d'Ivoire",
        "United States": "USA",
        "Iran": "IR Iran",
        "Cape Verde": "Cabo Verde",
        "DR Congo": "Congo DR",
    }
    df_hist["home_team"] = df_hist["home_team"].replace(name_mapping)
    df_hist["away_team"] = df_hist["away_team"].replace(name_mapping)

    print("Calculating dynamic ELO ratings across all matches...")
    elo_ratings = {}
    home_elos = []
    away_elos = []

    for idx, row in df_hist.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = elo_ratings.get(home, 1500.0)
        away_elo = elo_ratings.get(away, 1500.0)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        K = get_k_factor(row.get("tournament", ""))
        hs = row["home_score"]
        as_ = row["away_score"]

        if hs > as_:
            outcome_home = 1.0
        elif hs < as_:
            outcome_home = 0.0
        else:
            outcome_home = 0.5

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))

        elo_ratings[home] = home_elo + K * (outcome_home - expected_home)
        elo_ratings[away] = away_elo + K * ((1.0 - outcome_home) - (1.0 - expected_home))

    df_hist["home_elo"] = home_elos
    df_hist["away_elo"] = away_elos

    print("Preparing features...")
    features = []
    targets = []
    dates = []

    for idx, row in df_hist.iterrows():
        home_elo = row["home_elo"]
        away_elo = row["away_elo"]
        elo_diff = home_elo - away_elo

        neutral_raw = row.get("neutral", False)
        if isinstance(neutral_raw, str):
            neutral = 1 if neutral_raw.strip().lower() == "true" else 0
        else:
            neutral = 1 if neutral_raw else 0

        hs = row["home_score"]
        as_ = row["away_score"]

        if hs > as_:
            target = 2  # Home Win
        elif hs < as_:
            target = 0  # Away Win
        else:
            target = 1  # Draw

        features.append([elo_diff, neutral, home_elo, away_elo])
        targets.append(target)
        dates.append(row["date"])

    X = pd.DataFrame(features, columns=["elo_diff", "neutral", "home_elo", "away_elo"])
    y = pd.Series(targets)
    dates = pd.Series(dates)

    # ----------------------------------------------------------------
    # Train-Test Validation (Temporal Split at 2023-01-01)
    # ----------------------------------------------------------------
    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    split_date = pd.to_datetime("2023-01-01")
    train_idx = dates < split_date
    test_idx = dates >= split_date

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train samples (before 2023): {len(X_train)}")
    print(f"Test samples (2023+): {len(X_test)}")

    val_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    val_model.fit(X_train, y_train)

    y_pred = val_model.predict(X_test)
    y_prob = val_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    print(f"Test Set Accuracy: {acc*100:.2f}%")
    print(f"Test Set Log Loss: {loss:.4f}")
    print("\nClassification Report (Test Set):")
    target_names = ["Away Win (0)", "Draw (1)", "Home Win (2)"]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("------------------------------------\n")

    # ----------------------------------------------------------------
    # Retrain final model on 100% of data
    # ----------------------------------------------------------------
    print("Retraining final model on 100% of all data...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    print("Saving trained model assets...")
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    with open("team_elos.pkl", "wb") as f:
        pickle.dump(elo_ratings, f)

    print("Done! rf_model.pkl and team_elos.pkl saved successfully.")

if __name__ == "__main__":
    train_model()
