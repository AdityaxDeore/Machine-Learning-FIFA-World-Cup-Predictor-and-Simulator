import os
import math
import hashlib
import pickle
import warnings
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="FIFA World Cup 2026 Predictor & Simulator API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and datasets
def load_assets():
    base_dir = r"c:\Users\adity\OneDrive\Documents\FIFA 2026\FIFA-2026-Winner-Predictor-cum-Simulator"
    try:
        with open(os.path.join(base_dir, "rf_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(base_dir, "team_elos.pkl"), "rb") as f:
            team_elos = pickle.load(f)
        teams_df = pd.read_csv(os.path.join(base_dir, "teams.csv"))
        player_stats_df = pd.read_csv(os.path.join(base_dir, "player_stats.csv"))
        return rf_model, team_elos, teams_df, player_stats_df
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None, None, None

rf_model, team_elos, teams_df, player_stats_df = load_assets()

if rf_model is None:
    raise RuntimeError("Failed to load backend assets (model, elos, csv files).")

# Helper: goalscorer probabilities calculation
def get_player_goalscorer_probs(team_name, expected_goals):
    team_row = teams_df[teams_df["team_name"] == team_name]
    if team_row.empty:
        team_row = teams_df[teams_df["team_name"].str.lower() == team_name.lower()]
        if team_row.empty:
            return []
            
    t_id = team_row.iloc[0]["team_id"]
    players = player_stats_df[player_stats_df["team_id"] == t_id].copy()
    if players.empty:
        return []
        
    players["goals"] = players["goals"].fillna(0)
    players["shots_on_target"] = players["shots_on_target"].fillna(0)
    players["assists"] = players["assists"].fillna(0)
    
    position_baselines = {"FWD": 0.5, "MID": 0.2, "DEF": 0.05, "GK": 0.0}
    players["pos_baseline"] = players["position"].map(position_baselines).fillna(0.1)
    
    players["weight"] = (players["goals"] * 5.0 + 
                         players["shots_on_target"] * 1.5 + 
                         players["assists"] * 1.0 + 
                         players["pos_baseline"])
    
    total_weight = players["weight"].sum()
    if total_weight > 0:
        players["rel_weight"] = players["weight"] / total_weight
    else:
        players["rel_weight"] = 1.0 / len(players)
        
    players["score_prob"] = 1.0 - np.exp(-expected_goals * players["rel_weight"])
    top = players.sort_values(by="score_prob", ascending=False).head(3)
    return [{"name": row["player_name"], "prob": float(row["score_prob"]), "position": row["position"]} for idx, row in top.iterrows()]

class PredictRequest(BaseModel):
    team1: str
    team2: str
    neutral: bool = True

@app.get("/api/teams")
def get_teams():
    all_teams = sorted(list(team_elos.keys()))
    wc_teams = sorted(list(teams_df["team_name"].unique()))
    return {"all_teams": all_teams, "wc_teams": wc_teams}

@app.post("/api/predict")
def predict_match(req: PredictRequest):
    # Use ELO from dict, or fall back to 1500 for unknown teams
    elo1 = team_elos.get(req.team1, 1500.0)
    elo2 = team_elos.get(req.team2, 1500.0)

    elo_diff = elo1 - elo2
    neutral_val = 1 if req.neutral else 0
    
    # Predict Win/Draw/Loss probabilities
    probs = rf_model.predict_proba([[elo_diff, neutral_val, elo1, elo2]])[0]
    prob_away, prob_draw, prob_home = float(probs[0]), float(probs[1]), float(probs[2])
    
    # Determine the predicted outcome class: 0 = Away Win, 1 = Draw, 2 = Home Win
    pred_outcome = int(np.argmax(probs))
    
    # xG computation using ELO-based attack/defense ratings
    adj_diff = elo_diff
    if not req.neutral:
        adj_diff += 100.0
    avg_goals = 1.35
    exp_g1 = avg_goals * (10 ** (adj_diff / 400.0)) ** 0.35
    exp_g2 = avg_goals * (10 ** (-adj_diff / 400.0)) ** 0.35

    # --- Build full score probability grid (for chart top-5) ---
    score_probs = []
    for g1 in range(8):
        for g2 in range(8):
            p_h = np.exp(-exp_g1) * (exp_g1 ** g1) / math.factorial(g1)
            p_a = np.exp(-exp_g2) * (exp_g2 ** g2) / math.factorial(g2)
            score_probs.append((f"{g1} - {g2}", float(p_h * p_a)))
    score_probs = sorted(score_probs, key=lambda x: x[1], reverse=True)[:5]
    score_probs_list = [{"scoreline": s, "prob": p} for s, p in score_probs]

    # --- Smarter predicted scoreline via outcome-constrained Poisson mode ---
    # Find the single most probable scoreline matching the predicted outcome class
    best_score = (0, 0)
    max_p_constrained = -1
    
    for g1 in range(8):
        for g2 in range(8):
            is_match = False
            if pred_outcome == 2 and g1 > g2:
                is_match = True
            elif pred_outcome == 0 and g1 < g2:
                is_match = True
            elif pred_outcome == 1 and g1 == g2:
                is_match = True
                
            if is_match:
                p_h = np.exp(-exp_g1) * (exp_g1 ** g1) / math.factorial(g1)
                p_a = np.exp(-exp_g2) * (exp_g2 ** g2) / math.factorial(g2)
                p_joint = p_h * p_a
                if p_joint > max_p_constrained:
                    max_p_constrained = p_joint
                    best_score = (g1, g2)

    if max_p_constrained == -1:  # Fallback in case of absolute underflow
        g1r = max(0, round(exp_g1))
        g2r = max(0, round(exp_g2))
        if pred_outcome == 2:
            best_score = (max(g1r, g2r + 1), g2r)
        elif pred_outcome == 0:
            best_score = (g1r, max(g2r, g1r + 1))
        else:
            eq = round((exp_g1 + exp_g2) / 2)
            best_score = (eq, eq)

    
    # Top Scorers
    scorers1 = get_player_goalscorer_probs(req.team1, exp_g1)
    scorers2 = get_player_goalscorer_probs(req.team2, exp_g2)
    
    # Fetch team metadata profile
    t1_wc = teams_df[teams_df["team_name"] == req.team1]
    t2_wc = teams_df[teams_df["team_name"] == req.team2]
    metadata = {}
    if not t1_wc.empty and not t2_wc.empty:
        metadata = {
            "team1_rank": int(t1_wc.iloc[0]['fifa_ranking_pre_tournament']),
            "team2_rank": int(t2_wc.iloc[0]['fifa_ranking_pre_tournament']),
            "team1_manager": t1_wc.iloc[0]['manager_name'],
            "team2_manager": t2_wc.iloc[0]['manager_name'],
        }
            
    return {
        "team1_elo": int(elo1),
        "team2_elo": int(elo2),
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "exp_g1": float(exp_g1),
        "exp_g2": float(exp_g2),
        "predicted_score": best_score,
        "scoreline_probs": score_probs_list,
        "scorers1": scorers1,
        "scorers2": scorers2,
        "metadata": metadata
    }

# Simulation Helpers
def simulate_single_match_fast(t1, t2, elos_dict, is_knockout=True):
    elo1 = elos_dict.get(t1, 1500.0)
    elo2 = elos_dict.get(t2, 1500.0)
    elo_diff = elo1 - elo2
    
    probs = rf_model.predict_proba([[elo_diff, 1, elo1, elo2]])[0]
    prob_away, prob_draw, prob_home = probs[0], probs[1], probs[2]
    
    if is_knockout:
        p_win = prob_home / (prob_home + prob_away)
        p_lose = prob_away / (prob_home + prob_away)
        winner = np.random.choice([t1, t2], p=[p_win, p_lose])
        
        expected_1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        outcome_1 = 1.0 if winner == t1 else 0.0
        elos_dict[t1] = elo1 + 60 * (outcome_1 - expected_1)
        elos_dict[t2] = elo2 + 60 * ((1.0 - outcome_1) - (1.0 - expected_1))
        return winner
    else:
        outcome = np.random.choice([0, 1, 2], p=[prob_away, prob_draw, prob_home])
        outcome_1 = 1.0 if outcome == 2 else (0.5 if outcome == 1 else 0.0)
        expected_1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        elos_dict[t1] = elo1 + 40 * (outcome_1 - expected_1)
        elos_dict[t2] = elo2 + 40 * ((1.0 - outcome_1) - (1.0 - expected_1))
        return outcome

def simulate_match_bracket(t1, t2, elos_dict):
    elo1 = elos_dict.get(t1, 1500.0)
    elo2 = elos_dict.get(t2, 1500.0)
    elo_diff = elo1 - elo2
    
    probs = rf_model.predict_proba([[elo_diff, 1, elo1, elo2]])[0]
    prob_away, prob_draw, prob_home = probs[0], probs[1], probs[2]
    
    p_win = prob_home / (prob_home + prob_away)
    p_lose = prob_away / (prob_home + prob_away)
    
    avg_goals = 1.35
    exp_g1 = avg_goals * (10 ** (elo_diff / 400.0)) ** 0.35
    exp_g2 = avg_goals * (10 ** (-elo_diff / 400.0)) ** 0.35
    
    goals1 = int(np.random.poisson(exp_g1))
    goals2 = int(np.random.poisson(exp_g2))
    
    pen_win = None
    if goals1 == goals2:
        winner = np.random.choice([t1, t2], p=[p_win, p_lose])
        pen_win = winner
    else:
        winner = t1 if goals1 > goals2 else t2
        
    expected_1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    outcome_1 = 1.0 if winner == t1 else 0.0
    elos_dict[t1] = elo1 + 60 * (outcome_1 - expected_1)
    elos_dict[t2] = elo2 + 60 * ((1.0 - outcome_1) - (1.0 - expected_1))
    
    return winner, goals1, goals2, pen_win

@app.post("/api/simulate-bracket")
def simulate_bracket():
    elos_sim = team_elos.copy()
    
    # R16 Matchups
    r16_matchups = [
        ("Canada", "Morocco"),
        ("Paraguay", "France"),
        ("Brazil", "Norway"),
        ("Mexico", "England"),
        ("Portugal", "Spain"),
        ("USA", "Belgium"),
        ("Argentina", "Egypt"),
        ("Switzerland", "Colombia")
    ]
    
    r16_results = []
    qf_teams = []
    for t1, t2 in r16_matchups:
        if t1 == "Canada" and t2 == "Morocco":
            r16_results.append({"t1": t1, "t2": t2, "g1": 1, "g2": 2, "winner": "Morocco", "pen": None})
            qf_teams.append("Morocco")
        elif t1 == "Paraguay" and t2 == "France":
            r16_results.append({"t1": t1, "t2": t2, "g1": 0, "g2": 2, "winner": "France", "pen": None})
            qf_teams.append("France")
        else:
            w, g1, g2, pen = simulate_match_bracket(t1, t2, elos_sim)
            r16_results.append({"t1": t1, "t2": t2, "g1": g1, "g2": g2, "winner": w, "pen": pen})
            qf_teams.append(w)
            
    # QF
    qf_matchups = [(qf_teams[0], qf_teams[1]), (qf_teams[2], qf_teams[3]), (qf_teams[4], qf_teams[5]), (qf_teams[6], qf_teams[7])]
    qf_results = []
    sf_teams = []
    for t1, t2 in qf_matchups:
        w, g1, g2, pen = simulate_match_bracket(t1, t2, elos_sim)
        qf_results.append({"t1": t1, "t2": t2, "g1": g1, "g2": g2, "winner": w, "pen": pen})
        sf_teams.append(w)
        
    # SF
    sf_matchups = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
    sf_results = []
    final_teams = []
    for t1, t2 in sf_matchups:
        w, g1, g2, pen = simulate_match_bracket(t1, t2, elos_sim)
        sf_results.append({"t1": t1, "t2": t2, "g1": g1, "g2": g2, "winner": w, "pen": pen})
        final_teams.append(w)
        
    # Final
    champ, g1, g2, pen = simulate_match_bracket(final_teams[0], final_teams[1], elos_sim)
    final_result = {"t1": final_teams[0], "t2": final_teams[1], "g1": g1, "g2": g2, "winner": champ, "pen": pen}
    
    return {
        "r16": r16_results,
        "qf": qf_results,
        "sf": sf_results,
        "final": final_result,
        "champion": champ
    }

class MonteCarloRequest(BaseModel):
    runs: int = 5000

@app.post("/api/simulate-monte-carlo")
def simulate_monte_carlo(req: MonteCarloRequest):
    r16_teams = ["Canada", "Morocco", "Paraguay", "France", "Brazil", "Norway", "Mexico", "England", "Portugal", "Spain", "USA", "Belgium", "Argentina", "Egypt", "Switzerland", "Colombia"]
    team_reach = {t: {"R16": 0, "QF": 0, "SF": 0, "Final": 0, "Winner": 0} for t in r16_teams}
    
    for _ in range(req.runs):
        elos_sim = team_elos.copy()
        
        # R16
        r16_winners = {
            "Canada/Morocco": "Morocco",
            "Paraguay/France": "France",
            "Brazil/Norway": simulate_single_match_fast("Brazil", "Norway", elos_sim),
            "Mexico/England": simulate_single_match_fast("Mexico", "England", elos_sim),
            "Portugal/Spain": simulate_single_match_fast("Portugal", "Spain", elos_sim),
            "USA/Belgium": simulate_single_match_fast("USA", "Belgium", elos_sim),
            "Argentina/Egypt": simulate_single_match_fast("Argentina", "Egypt", elos_sim),
            "Switzerland/Colombia": simulate_single_match_fast("Switzerland", "Colombia", elos_sim)
        }
        for t in r16_teams:
            team_reach[t]["R16"] += 1
            
        # QF
        qf_winners = {
            "Morocco/France": simulate_single_match_fast(r16_winners["Canada/Morocco"], r16_winners["Paraguay/France"], elos_sim),
            "Brazil/Norway/Mexico/England": simulate_single_match_fast(r16_winners["Brazil/Norway"], r16_winners["Mexico/England"], elos_sim),
            "Portugal/Spain/USA/Belgium": simulate_single_match_fast(r16_winners["Portugal/Spain"], r16_winners["USA/Belgium"], elos_sim),
            "Argentina/Egypt/Switzerland/Colombia": simulate_single_match_fast(r16_winners["Argentina/Egypt"], r16_winners["Switzerland/Colombia"], elos_sim)
        }
        for t in qf_winners.values():
            team_reach[t]["QF"] += 1
            
        # SF
        sf_winners = {
            "Morocco/France/Brazil/Norway/Mexico/England": simulate_single_match_fast(qf_winners["Morocco/France"], qf_winners["Brazil/Norway/Mexico/England"], elos_sim),
            "Portugal/Spain/USA/Belgium/Argentina/Egypt/Switzerland/Colombia": simulate_single_match_fast(qf_winners["Portugal/Spain/USA/Belgium"], qf_winners["Argentina/Egypt/Switzerland/Colombia"], elos_sim)
        }
        for t in sf_winners.values():
            team_reach[t]["SF"] += 1
            
        # Final
        winner = simulate_single_match_fast(
            sf_winners["Morocco/France/Brazil/Norway/Mexico/England"], 
            sf_winners["Portugal/Spain/USA/Belgium/Argentina/Egypt/Switzerland/Colombia"], 
            elos_sim
        )
        for t in sf_winners.values():
            team_reach[t]["Final"] += 1
        team_reach[winner]["Winner"] += 1
        
    # Scale to percentage
    results = []
    for team, stages in team_reach.items():
        results.append({
            "team": team,
            "R16": round((stages["R16"] / req.runs) * 100, 2),
            "QF": round((stages["QF"] / req.runs) * 100, 2),
            "SF": round((stages["SF"] / req.runs) * 100, 2),
            "Final": round((stages["Final"] / req.runs) * 100, 2),
            "Winner": round((stages["Winner"] / req.runs) * 100, 2)
        })
        
    # Sort by winner odds
    results = sorted(results, key=lambda x: x["Winner"], reverse=True)
    return results

@app.get("/api/standings")
def get_standings(group: str = None, confederation: str = None):
    team_stats_list = []
    for idx, row in teams_df.iterrows():
        name = row["team_name"]
        
        # Filters
        if group and group != "All Groups" and row["group_letter"] != group:
            continue
        if confederation and confederation != "All Confederations" and row["confederation"] != confederation:
            continue
            
        lookup_name = name.replace("Turkey", "Türkiye").replace("Ivory Coast", "Côte d'Ivoire").replace("Bosnia & Herzegovina", "Bosnia and Herzegovina")
        cur_elo = team_elos.get(name, team_elos.get(lookup_name, row["elo_rating"]))
        diff = cur_elo - row["elo_rating"]
        
        team_stats_list.append({
            "team": name,
            "group": row["group_letter"],
            "confederation": row["confederation"],
            "manager": row["manager_name"],
            "fifa_rank": int(row["fifa_ranking_pre_tournament"]),
            "base_elo": int(row["elo_rating"]),
            "current_elo": int(cur_elo),
            "gain_loss": f"+{int(diff)}" if diff >= 0 else f"{int(diff)}"
        })
        
    team_stats_list = sorted(team_stats_list, key=lambda x: x["current_elo"], reverse=True)
    return team_stats_list

# Serve static frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
