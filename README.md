

# 🎮 VALORANT Personal Coach

## Personalized Map-Based Agent Recommendation Using Machine Learning

VALORANT Personal Coach is a personalized machine-learning project designed to
recommend the **Top 5–6 VALORANT agents** for an individual player after the
current map is known.

Instead of using generic agent tier lists or global game statistics, the system
learns from a player's own Competitive match history.

The recommendation is based on historical factors such as:

- Previous performance with an agent
- Previous performance on a map
- Agent × Map performance
- Recent gameplay form
- Agent familiarity
- Agent usage frequency
- Agent recency
- Historical K/D and ACS trends

The final objective is to provide a ranked list of agents that are personally
suitable for the player on the selected map.

---

# 🎯 Project Objective

Given a VALORANT map, the system evaluates the player's previously used agents
and generates a ranked **Top 5–6 personalized agent recommendation list**.

The machine-learning models estimate:

`P(Win | Map, Candidate Agent, Historical Player Features)`

The predicted probabilities are then used as recommendation scores for ranking
candidate agents.

Only information available **before the target match** is used for model input.

Current-match statistics such as kills, deaths, K/D, ACS, DDΔ, TRS, final score,
and match-ending type are not used directly as predictive features because they
would introduce data leakage.

---

# 👥 Player-Specific Datasets

The project currently uses two independently maintained VALORANT Competitive
match-history datasets.

## Tanishka Dataset

- Total Matches: **297**
- Binary modelling matches: **287**
- Draws: **10**
- Main Agent: Sage
- Personal agent pool contains 9 historically used agents

## Dev Dataset

- Total Matches: **1080**
- Binary modelling matches: **1055**
- Draws: **25**
- Main Agent: Neon
- Personal agent pool contains 26 historically used agents

The two datasets are **not merged**.

The same data-cleaning, feature-engineering, and machine-learning methodology is
applied independently to each player.

Therefore:

```text
Tanishka History
        ↓
Tanishka Features
        ↓
Tanishka Models
        ↓
Tanishka Recommendations

and



Dev History
        ↓
Dev Features
        ↓
Dev Models
        ↓
Dev Recommendations

One player's match history never contributes to the other player's model.



This allows the system to learn separate gameplay profiles and potentially

recommend different agents to different players for the same map.

🗂️ Project Structure



valorant_personal_coach/
│
├── data/
│   ├── competitive_matches.xlsx
│   ├── competitive_matches_dev.xlsx
│   │
│   └── processed/
│       ├── competitive_matches_cleaned.csv
│       ├── competitive_matches_dev_cleaned.csv
│       ├── competitive_matches_features.csv
│       └── competitive_matches_dev_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── DEVnotebooks/
│   ├── DEV_01_data_exploration.ipynb
│   └── DEV_02_model_training.ipynb
│
├── src/
│   ├── clean_data.py
│   └── feature_engineering.py
│
├── figures/
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore

🏆 Agent Recommendation System

At recommendation time, the player provides the current map.

Example:

Map = Haven

The system constructs one candidate profile for every agent the player has
previously used.

Haven + Sage
Haven + Killjoy
Haven + Clove
Haven + Breach
...

For every candidate agent, the system calculates the player's historical
pre-match features.

The selected model then estimates a recommendation score:

Candidate Agent
      ↓
Historical Player Features
      ↓
ML Model
      ↓
Predicted Win Probability

The candidates are sorted from highest to lowest predicted probability.

The final output is:

1. Agent A
2. Agent B
3. Agent C
4. Agent D
5. Agent E
6. Agent F

This produces the player's personalized Top 5–6 agent recommendations for
the selected map.