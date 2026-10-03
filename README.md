# 🎮 VALORANT Personal Coach

## Personalized Map-Based, Role-Aware Agent Recommendation Using Machine Learning

**VALORANT Personal Coach** is a personalized machine-learning project that
recommends VALORANT agents using an individual player's own Competitive match
history.

Instead of relying on generic tier lists or global statistics, the system learns
a separate gameplay profile for each player and uses the current map together
with historical player information to estimate personalized agent suitability.

The final recommendation layer is **role-aware**. Rather than returning one
overall Top-5 or Top-6 list that may be dominated by a single role, the system
can return up to **two recommended agents per role**:

- Duelist
- Controller
- Initiator
- Sentinel

The project currently contains two completely separate player pipelines:
**Tanishka** and **Dev**.

---

# 🎯 Project Objective

Given a VALORANT map, the system creates a historical candidate profile for
each agent the player has previously used and estimates:

`P(Win | Map, Candidate Agent, Role, Historical Player Features)`

The predicted win probability is used as the recommendation score.

Agents are then ranked **within their respective roles**, allowing the final
system to return up to **two personalized candidate agents per role**.

Only information that would have been available **before the target match** is
used as model input.

Current-match statistics such as kills, deaths, K/D, ACS, DDΔ, final score, and
match-ending type are not used directly as pre-match predictors because doing so
would introduce **data leakage**.

Instead, historical versions of relevant statistics are calculated using only
matches that occurred before each target match.

---

# 🔄 Updated Role-Aware Dataset Design

The original datasets contained `TRS (Tracker Score)`. During project
refinement, the data design was updated:

- `TRS` was removed from the revised datasets and final modelling pipeline.
- A categorical `Role` feature was added immediately after `Agent`.
- Every agent is mapped to one official VALORANT role:
  - Duelist
  - Controller
  - Initiator
  - Sentinel
- **The selected agent determines the official role.**
- ACS does **not** determine role.
- Historical ACS is retained as a performance signal and is also compared with
  the player's historical ACS baseline for that role.

One role-aware feature used by the pipeline is:

`agent_role_acs_delta = agent_previous_acs - role_previous_acs`

This measures how the player's historical ACS with a specific agent compares
with their historical ACS baseline for that agent's role.

---

# 👥 Player-Specific Datasets

The project uses two independently maintained VALORANT Competitive match-history
datasets.

## Tanishka Dataset

- Total matches: **304**
- Binary modelling matches: **294**
- Draws: **10**
- Historically used agents: **9**
- Most-played agent: **Sage**
- Dominant role: **Sentinel**
- Role distribution:
  - Sentinel: **205 matches**
  - Controller: **65 matches**
  - Initiator: **31 matches**
  - Duelist: **3 matches**

## Dev Dataset

- Total matches: **1080**
- Binary modelling matches: **1055**
- Draws: **25**
- Historically used agents: **26**
- Most-played agent: **Neon**
- Dominant role: **Duelist**
- Role distribution:
  - Duelist: **885 matches**
  - Controller: **98 matches**
  - Sentinel: **69 matches**
  - Initiator: **28 matches**

The two datasets are **never merged**.

```text
Tanishka History
        ↓
Tanishka Cleaning
        ↓
Tanishka Historical Features
        ↓
Tanishka Models
        ↓
Tanishka Recommendations


Dev History
        ↓
Dev Cleaning
        ↓
Dev Historical Features
        ↓
Dev Models
        ↓
Dev Recommendations
```

One player's match history never contributes to the other player's model. This
allows the system to learn distinct gameplay profiles and produce different
recommendations for different players on the same map.

---

# 🧹 Data Cleaning Pipeline

`src/clean_data.py` reads the revised Excel files from `data/updated/`:

- `competitive_matches_role_updated.xlsx`
- `competitive_matches_dev_role_updated.xlsx`

The cleaning pipeline:

- validates required columns,
- validates the Agent → Role mapping,
- verifies result values,
- handles chronological dates,
- recomputes Win Flag,
- recomputes Score Margin and Rounds Played,
- recomputes K/D and per-round statistics,
- identifies deathless matches,
- classifies match-ending type,
- identifies surrendering side,
- marks binary-model eligibility,
- sorts matches chronologically,
- and performs data-quality checks.

The cleaned datasets are saved back into `data/updated/`.

---

# 🧠 Leakage-Free Historical Feature Engineering

`src/feature_engineering.py` transforms the cleaned datasets into chronological,
player-specific historical features.

For every target match, only information from **earlier matches** is used.

Examples include:

## Overall Historical Features

- `previous_matches`
- `previous_win_rate`
- `previous_kd`

## Recent Form

- `recent_5_win_rate`
- `recent_10_win_rate`
- `recent_5_kd`
- `recent_10_kd`
- `recent_5_acs`
- `recent_10_acs`

## Agent History

- `agent_previous_matches`
- `agent_previous_win_rate`
- `agent_previous_kd`
- `agent_previous_acs`
- `agent_usage_frequency`
- `agent_recency`
- `agent_familiarity`

## Map History

- `map_previous_matches`
- `map_previous_win_rate`
- `map_previous_kd`

## Agent × Map History

- `agent_map_previous_matches`
- `agent_map_previous_win_rate`
- `agent_map_previous_kd`

## Role-Aware History

- `role_previous_matches`
- `role_previous_win_rate`
- `role_previous_kd`
- `role_previous_acs`
- `role_usage_frequency`
- `agent_role_acs_delta`

The generated feature datasets currently contain **36 columns** for each player.

---

# 🤖 Machine-Learning Models

Three substantially different classification algorithms are trained and
evaluated independently for each player:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **CatBoost Classifier**

The categorical model inputs include:

- Map
- Agent
- Role
- Agent familiarity

Logistic Regression and Random Forest use preprocessing with one-hot encoding,
while CatBoost can work with categorical features in its own modelling pipeline.

Hyperparameter tuning and model selection are performed using chronological
training and validation data. The held-out test period is kept separate from
model selection.

---

# 📊 Evaluation

The project uses retrospective ranking metrics:

- **MRR (Mean Reciprocal Rank)**
- **Hit@5**
- **Hit@6**

These metrics evaluate where the agent that was historically selected appears
inside the model's raw candidate ranking.

They are **not the same as the final recommendation display**.

The final user-facing recommendation layer is role-aware and returns up to
**two agents per role**.

This distinction is important:

```text
Raw model ranking
      ↓
Retrospective Hit@5 / Hit@6 / MRR evaluation
      ↓
Role-aware ranking
      ↓
Top 2 candidate agents within each role
```

---

# 🏁 Current Final Results

## Tanishka

Final selected model: **CatBoost**

Held-out test performance:

- Evaluated matches: **45**
- MRR: **0.1934**
- Hit@5: **26.67%**
- Hit@6: **44.44%**

Recommendation runtime across the tested maps:

- Average: approximately **0.048 seconds**
- Maximum: approximately **0.073 seconds**

## Dev

Final selected model: **Random Forest**

Held-out test performance:

- Evaluated matches: **158**
- MRR: **0.3576**
- Hit@5: **70.25%**
- Hit@6: **78.48%**

Recommendation runtime across the tested maps:

- Average: approximately **0.192 seconds**
- Maximum: approximately **0.228 seconds**

Both recommendation pipelines run well within the project's approximate
**10–20 second response-time target**.

---

# 🏆 Role-Aware Agent Recommendation System

At recommendation time, the player provides the current map.

Example:

```text
Map = Haven
```

For every historically used candidate agent, the system constructs a profile
containing the player's historical agent, map, Agent × Map, recent-form, and
role-level features.

```text
Selected Map + Candidate Agent + Role
                ↓
Historical Player Features
                ↓
Selected ML Model
                ↓
Predicted Win Probability
                ↓
Rank Within Role
                ↓
Up to Top 2 Agents per Role
```

## Example: Tanishka on Haven

The current saved result returns:

| Role | Rank | Agent |
|---|---:|---|
| Sentinel | 1 | Killjoy |
| Sentinel | 2 | Sage |
| Controller | 1 | Astra |
| Controller | 2 | Clove |
| Initiator | 1 | Breach |
| Initiator | 2 | Skye |
| Duelist | 1 | Reyna |

Only one Duelist is returned because Tanishka's recorded Duelist history
contains only one historically used candidate agent.

## Example: Dev on Haven

The current saved result returns:

| Role | Rank | Agent |
|---|---:|---|
| Duelist | 1 | Phoenix |
| Duelist | 2 | Jett |
| Controller | 1 | Brimstone |
| Controller | 2 | Clove |
| Sentinel | 1 | Chamber |
| Sentinel | 2 | Killjoy |
| Initiator | 1 | Breach |
| Initiator | 2 | Gekko |

These examples demonstrate why the system is player-specific: the same map can
produce different recommendations for different players.

---

# 🗂️ Project Structure

Simplified structure of the current project:

```text
valorant_personal_coach/
│
├── data/
│   ├── competitive_matches.xlsx
│   ├── competitive_matches_dev.xlsx
│   │
│   └── updated/
│       ├── competitive_matches_role_updated.xlsx
│       ├── competitive_matches_dev_role_updated.xlsx
│       ├── competitive_matches_cleaned.csv
│       ├── competitive_matches_dev_cleaned.csv
│       ├── competitive_matches_features.csv
│       └── competitive_matches_dev_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training_role_updated.ipynb
│
├── DEVnotebooks/
│   ├── DEV_01_data_exploration2.ipynb
│   └── DEV_02_model_training_role_updated.ipynb
│
├── src/
│   ├── clean_data.py
│   └── feature_engineering.py
│
├── figures/
│
├── results/
│   ├── tanishka_hyperparameter_tuning.csv
│   ├── tanishka_validation_results.csv
│   ├── tanishka_final_test_results.csv
│   ├── tanishka_feature_importance.csv
│   ├── tanishka_runtime_results.csv
│   ├── tanishka_haven_recommendations.csv
│   ├── dev_hyperparameter_tuning.csv
│   ├── dev_validation_results.csv
│   ├── dev_final_test_results.csv
│   ├── dev_feature_importance.csv
│   ├── dev_runtime_results.csv
│   └── dev_haven_recommendations.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

Generated CatBoost training-log directories and Git metadata are omitted from the
simplified tree above.

---

# ▶️ How to Run the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses:

- pandas
- numpy
- matplotlib
- openpyxl
- Jupyter
- scikit-learn
- CatBoost
- joblib

## 2. Clean the Revised Datasets

From the project root:

```bash
python src/clean_data.py
```

This generates:

```text
data/updated/competitive_matches_cleaned.csv
data/updated/competitive_matches_dev_cleaned.csv
```

## 3. Generate Historical Features

```bash
python src/feature_engineering.py
```

This generates:

```text
data/updated/competitive_matches_features.csv
data/updated/competitive_matches_dev_features.csv
```

## 4. Run Exploratory Analysis

Tanishka:

```text
notebooks/01_data_exploration.ipynb
```

Dev:

```text
DEVnotebooks/DEV_01_data_exploration2.ipynb
```

## 5. Run Model Training and Recommendation

Tanishka:

```text
notebooks/02_model_training_role_updated.ipynb
```

Dev:

```text
DEVnotebooks/DEV_02_model_training_role_updated.ipynb
```

Running the modelling notebooks regenerates the corresponding files in
`results/`.

---

# ⚠️ Important Modelling Notes

- The system is **personalized**, not a universal VALORANT tier list.
- Tanishka and Dev are trained completely independently.
- Role is determined by **agent identity**, not ACS.
- ACS is used only through historical information when making pre-match
  recommendations.
- `TRS` belongs only to the original EDA history and is not part of the revised
  modelling pipeline.
- Chronological feature engineering is used to reduce temporal leakage.
- A player's sparse history with an agent, role, or Agent × Map combination may
  make some recommendation estimates less reliable.
- The historical dataset records only the result for the agent that was actually
  selected. It does not contain the counterfactual outcome for agents that were
  not selected.

Therefore, the system estimates **personal historical suitability** rather than
proving which agent would objectively have been the optimal choice in a match.

---

# 🚀 Future Improvements

Possible extensions include:

- Team-composition awareness
- Enemy-composition awareness
- Patch-specific balance information
- Professional meta trends
- Additional map-context features
- Larger personal match histories
- More robust evaluation for sparse roles and rarely played agents
- Real-time integration with an agent-select interface

---

# 📌 Final Outcome

The project demonstrates a complete end-to-end machine-learning workflow that
converts personal VALORANT Competitive match history into a **role-aware,
player-specific agent recommendation system**.

For a supplied map, the final system uses historical information to estimate
candidate-agent win probabilities and returns up to **two personalized agents
per role**, allowing different players to receive different recommendations for
the same map.
