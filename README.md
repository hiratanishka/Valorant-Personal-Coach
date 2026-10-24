# 🎮 VALORANT Personal Coach

## Personalized Map-Based, Role-Aware Agent Recommendation Using Machine Learning

**VALORANT Personal Coach** is a personalized machine-learning project that recommends VALORANT agents using an individual player's own Competitive match history.

Instead of relying on generic tier lists or global statistics, the system learns a separate gameplay profile for each player and uses the current map together with historical player information to estimate personalized agent suitability.

The final recommendation layer is **role-aware**. Rather than returning one overall Top-5 or Top-6 list that may be dominated by a single role, the system returns up to **two recommended agents per role**:

- Duelist
- Controller
- Initiator
- Sentinel

The project currently contains three completely separate player pipelines:

- **Tanishka**
- **Dev**
- **Anuj**

The current architecture is also designed so that additional standardized player datasets can be added later without changing the core cleaning and feature-engineering logic.

---

# 🎯 Project Objective

Given a VALORANT map, the system creates a historical candidate profile for each agent the selected player has previously used and estimates:

`P(Win | Map, Candidate Agent, Role, Historical Player Features)`

The predicted win probability is used as the recommendation score.

Agents are then ranked **within their respective roles**, allowing the final system to return up to **two personalized candidate agents per role**.

Only information that would have been available **before the target match** is used as model input.

Current-match statistics such as kills, deaths, K/D, ACS, DDΔ, final score, and match-ending type are not used directly as pre-match predictors because doing so would introduce **data leakage**.

Instead, historical versions of relevant performance statistics are calculated using only matches that occurred before each target match.

---

# 🔄 Role-Aware Dataset Design

The original raw Excel datasets contain match-level statistics, including `TRS (Tracker Score)`.

During preprocessing:

- `TRS` is removed from the final modelling pipeline.
- A categorical `Role` feature is generated from the selected `Agent`.
- Every supported agent is mapped to one official VALORANT role:
  - Duelist
  - Controller
  - Initiator
  - Sentinel
- **Agent identity determines Role.**
- ACS does **not** determine Role.
- ACS is retained as a historical performance signal.
- Current-match ACS is not used directly as a pre-match feature.

One role-aware feature used by the project is:

`agent_role_acs_delta = agent_previous_acs - role_previous_acs`

This compares the player's historical ACS with a specific agent against the player's historical ACS baseline for that agent's role.

---

# 👥 Player-Specific Datasets

The three player histories are kept completely separate.

## Tanishka Dataset

- Total matches: **304**
- Binary modelling matches: **294**
- Draws: **10**
- Historically used agents: **9**
- Most-played agent: **Sage**
- Dominant role: **Sentinel**
- Role distribution:
  - Sentinel: **205**
  - Controller: **65**
  - Initiator: **31**
  - Duelist: **3**

## Dev Dataset

- Total matches: **1080**
- Binary modelling matches: **1055**
- Draws: **25**
- Historically used agents: **26**
- Most-played agent: **Neon**
- Dominant role: **Duelist**
- Role distribution:
  - Duelist: **885**
  - Controller: **98**
  - Sentinel: **69**
  - Initiator: **28**

## Anuj Dataset

- Total matches: **715**
- Binary modelling matches: **706**
- Draws: **9**
- Historically used agents: **19**
- Most-played agent: **Cypher**
- Dominant role: **Sentinel**
- Role distribution:
  - Sentinel: **429**
  - Controller: **168**
  - Duelist: **73**
  - Initiator: **45**

The datasets are **never merged**.

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


Anuj History
        ↓
Anuj Cleaning
        ↓
Anuj Historical Features
        ↓
Anuj Models
        ↓
Anuj Recommendations
```

One player's match history never contributes to another player's model.

This allows the system to learn different gameplay profiles and produce different recommendations for different players on the same map.

---

# 🧹 Data Cleaning Pipeline

`src/clean_data.py` processes the raw Excel datasets stored directly inside `data/`.

Current raw files:

```text
data/competitive_matches.xlsx
data/competitive_matches_dev.xlsx
data/competitive_matches_anuj.xlsx
```

The cleaning pipeline:

- automatically handles the configured player datasets,
- detects the real header row,
- normalizes column names,
- removes `TRS` from the modelling dataset,
- derives `Role` from the official Agent → Role mapping,
- validates required columns,
- validates supported agents,
- verifies result values,
- converts dates and numerical fields,
- checks duplicate rows and duplicate Match IDs,
- recomputes Win Flag,
- recomputes Score Margin and Rounds Played,
- recomputes K/D and per-round statistics,
- identifies deathless matches,
- classifies match-ending type,
- identifies the surrendering side,
- marks binary-model eligibility,
- sorts matches chronologically,
- and performs data-quality checks.

The cleaned datasets are saved to:

```text
data/datasets/
├── competitive_matches_tanishka_cleaned.csv
├── competitive_matches_dev_cleaned.csv
└── competitive_matches_anuj_cleaned.csv
```

---

# 🧠 Leakage-Free Historical Feature Engineering

`src/feature_engineering.py` transforms each cleaned dataset into chronological, player-specific historical features.

For every target match, information is calculated using only matches that happened **before** that target match.

The generated feature datasets contain **36 columns**.

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

## Role-Aware History

- `role_previous_matches`
- `role_previous_win_rate`
- `role_previous_kd`
- `role_previous_acs`
- `role_usage_frequency`
- `agent_role_acs_delta`

## Map History

- `map_previous_matches`
- `map_previous_win_rate`
- `map_previous_kd`

## Agent × Map History

- `agent_map_previous_matches`
- `agent_map_previous_win_rate`
- `agent_map_previous_kd`

Generated feature files:

```text
data/datasets/
├── competitive_matches_tanishka_features.csv
├── competitive_matches_dev_features.csv
└── competitive_matches_anuj_features.csv
```

---

# 🤖 Machine-Learning Models

Three substantially different classification algorithms are trained and evaluated independently for every player:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **CatBoost Classifier**

Each player's notebook folder contains one notebook for each algorithm.

The categorical model inputs include:

- Map
- Agent
- Role
- Agent familiarity

The remaining model inputs are leakage-free numerical historical features.

Logistic Regression and Random Forest use preprocessing with one-hot encoding. CatBoost handles categorical variables through its own modelling pipeline.

---

# ⏳ Chronological Training Strategy

The project uses chronological splitting instead of a random train-test split.

The modelling notebooks use approximately:

```text
70% Training
15% Validation
15% Test
```

The earliest matches are used for training, the following period is used for validation and hyperparameter selection, and the most recent period is kept as the held-out test set.

This better represents the real use case:

```text
Past matches
      ↓
Learn player behaviour
      ↓
Predict / recommend for later matches
```

The held-out test set is not used for model selection.

---

# ⚙️ Hyperparameter Tuning

Each modelling notebook performs algorithm-specific hyperparameter tuning using chronological training and validation data.

The project primarily compares ranking performance using:

- **Hit@6**
- **MRR**

The selected configuration is then fitted on the appropriate historical training data before final held-out evaluation.

Because the models have different structures, their tuning spaces are intentionally different.

---

# 📊 Evaluation

The project uses retrospective ranking metrics:

- **MRR (Mean Reciprocal Rank)**
- **Hit@5**
- **Hit@6**

These metrics evaluate where the agent that was historically selected appears inside the model's raw candidate ranking.

They are **not the same as the final user-facing recommendation display**.

The final recommendation layer is role-aware and returns up to **two agents per role**.

```text
Raw model ranking
      ↓
Retrospective Hit@5 / Hit@6 / MRR
      ↓
Role-aware ranking
      ↓
Top 2 candidate agents within each role
```

---

# 🏆 Role-Aware Agent Recommendation System

At recommendation time, the player supplies the current map.

Example:

```text
Map = Haven
```

For every historically used candidate agent, the system creates a candidate profile containing the player's historical:

- overall performance,
- recent form,
- agent performance,
- role performance,
- map performance,
- Agent × Map performance,
- agent usage,
- agent recency,
- and agent familiarity.

The model then estimates the candidate's personalized win probability.

```text
Selected Map
      ↓
Candidate Agent
      ↓
Official Role
      ↓
Historical Player Features
      ↓
Machine-Learning Model
      ↓
Predicted Win Probability
      ↓
Rank Agents Within Each Role
      ↓
Top 2 Agents per Role
```

This means two different players can receive different recommendations for the same map.

---

# 📁 Model Output Files

Each model notebook stores its outputs inside `results/`.

The naming convention is:

```text
<player>_<model>_hyperparameter_tuning.csv
<player>_<model>_validation_results.csv
<player>_<model>_final_test_results.csv
<player>_<model>_feature_importance.csv
<player>_<model>_runtime_results.csv
<player>_<model>_haven_recommendations.csv
```

Examples:

```text
dev_random_forest_final_test_results.csv
anuj_catboost_feature_importance.csv
dev_logistic_regression_haven_recommendations.csv
```

Keeping the player name and model name in every output filename prevents one experiment from overwriting another.

Final numerical results are intentionally read from the generated CSV outputs rather than hard-coded into this README, because rerunning the pipeline or adding new datasets can change the metrics.

---

# 🗂️ Current Project Structure

```text
valorant_personal_coach/
│
├── data/
│   ├── competitive_matches.xlsx
│   ├── competitive_matches_dev.xlsx
│   ├── competitive_matches_anuj.xlsx
│   │
│   └── datasets/
│       ├── competitive_matches_tanishka_cleaned.csv
│       ├── competitive_matches_tanishka_features.csv
│       ├── competitive_matches_dev_cleaned.csv
│       ├── competitive_matches_dev_features.csv
│       ├── competitive_matches_anuj_cleaned.csv
│       └── competitive_matches_anuj_features.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   └── 03_catboost.ipynb
│
├── DEVnotebooks/
│   ├── EDA.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   └── 03_catboost.ipynb
│
├── ANUJnotebooks/
│   ├── EDA.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   └── 03_catboost.ipynb
│
├── src/
│   ├── clean_data.py
│   └── feature_engineering.py
│
├── results/
│   └── generated model-specific CSV outputs
│
├── README.md
├── requirements.txt
└── .gitignore
```

`.git/`, `.vscode/`, notebook checkpoints, CatBoost training logs, and other local metadata are omitted from the simplified project tree.

---

# ▶️ How to Run the Project

## 1. Install Dependencies

From the project root:

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

---

## 2. Clean All Configured Player Datasets

```bash
python src/clean_data.py
```

This generates:

```text
data/datasets/competitive_matches_tanishka_cleaned.csv
data/datasets/competitive_matches_dev_cleaned.csv
data/datasets/competitive_matches_anuj_cleaned.csv
```

---

## 3. Generate Leakage-Free Historical Features

```bash
python src/feature_engineering.py
```

This generates:

```text
data/datasets/competitive_matches_tanishka_features.csv
data/datasets/competitive_matches_dev_features.csv
data/datasets/competitive_matches_anuj_features.csv
```

---

## 4. Run Exploratory Data Analysis

### Tanishka

```text
notebooks/EDA.ipynb
```

### Dev

```text
DEVnotebooks/EDA.ipynb
```

### Anuj

```text
ANUJnotebooks/EDA.ipynb
```

---

## 5. Run Logistic Regression

```text
notebooks/01_logistic_regression.ipynb
DEVnotebooks/01_logistic_regression.ipynb
ANUJnotebooks/01_logistic_regression.ipynb
```

---

## 6. Run Random Forest

```text
notebooks/02_random_forest.ipynb
DEVnotebooks/02_random_forest.ipynb
ANUJnotebooks/02_random_forest.ipynb
```

---

## 7. Run CatBoost

```text
notebooks/03_catboost.ipynb
DEVnotebooks/03_catboost.ipynb
ANUJnotebooks/03_catboost.ipynb
```

Running a model notebook regenerates that player-model combination's files inside `results/`.

---

# 🔧 Expected Dataset Configuration

Each player-specific model notebook should point to the standardized files inside `data/datasets/`.

## Tanishka

```python
PLAYER_NAME = "Tanishka"
RESULT_PREFIX = "tanishka"

CLEANED_FILENAME = "competitive_matches_tanishka_cleaned.csv"
FEATURE_FILENAME = "competitive_matches_tanishka_features.csv"
```

## Dev

```python
PLAYER_NAME = "Dev"
RESULT_PREFIX = "dev"

CLEANED_FILENAME = "competitive_matches_dev_cleaned.csv"
FEATURE_FILENAME = "competitive_matches_dev_features.csv"
```

## Anuj

```python
PLAYER_NAME = "Anuj"
RESULT_PREFIX = "anuj"

CLEANED_FILENAME = "competitive_matches_anuj_cleaned.csv"
FEATURE_FILENAME = "competitive_matches_anuj_features.csv"
```

The model notebooks should use:

```python
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
```

The old `data/updated/` path is no longer part of the current project structure.

---

# ⚠️ Important Modelling Notes

- The system is **personalized**, not a universal VALORANT tier list.
- Tanishka, Dev, and Anuj are trained independently.
- Player histories are never pooled.
- Role is determined by **agent identity**, not ACS.
- `TRS` is not used in the final modelling pipeline.
- Current-match ACS, K/D, kills, deaths, DDΔ, score, and other post-match values are not direct pre-match inputs.
- Historical ACS is allowed because it is computed only from earlier matches.
- Chronological feature engineering is used to reduce temporal leakage.
- Chronological train/validation/test splitting is used instead of random splitting.
- Sparse history for a particular agent, role, map, or Agent × Map combination can make some recommendation estimates less reliable.
- The historical dataset records the outcome only for the agent that was actually selected.
- It does not contain the counterfactual outcome for agents that were not selected.

Therefore, the project estimates **personal historical suitability**, not a guaranteed objectively optimal agent choice.

---

# ➕ Adding More Player Datasets

The project is designed to support more player histories.

For an additional player:

1. Place the raw Excel file inside `data/`.
2. Add the player to the dataset configuration in `src/clean_data.py`.
3. Add the corresponding player entry in `src/feature_engineering.py`.
4. Run both scripts to generate the standardized cleaned and feature CSVs.
5. Create or configure the player's EDA and model notebooks to point to the generated files.
6. Use a unique `RESULT_PREFIX` so outputs remain separate.

The same methodology can therefore be extended beyond the current three players.

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
- Automated selection of the strongest model per player
- Real-time integration with an agent-select interface

---

# 📌 Final Outcome

The project demonstrates a complete end-to-end machine-learning workflow that converts personal VALORANT Competitive match history into a **role-aware, player-specific agent recommendation system**.

For a supplied map, the system uses leakage-free historical information to estimate candidate-agent win probabilities and returns up to **two personalized agents per role**.

Because every player is trained independently, the same map can produce different recommendations for Tanishka, Dev, Anuj, or any future player added to the pipeline.
