from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "datasets"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Keep this list aligned with clean_data.py.
# Add Player 4 and Player 5 here later.

DATASETS = [
    {
        "player_name": "Tanishka",
        "output_slug": "tanishka",
    },
    {
        "player_name": "Dev",
        "output_slug": "dev",
    },
    {
        "player_name": "Anuj",
        "output_slug": "anuj",
    },
]


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Divide pandas Series while avoiding division by zero.
    """

    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    input_path,
    output_path,
    player_name,
):
    """
    Create leakage-free historical features for one player's
    VALORANT Competitive match history.

    Every feature for Match N uses only matches that happened
    BEFORE Match N.

    Role is known before the match because it is determined by
    the selected agent.

    ACS is never taken from the current match as an input.
    Only historical ACS summaries are used.
    """

    print("\n" + "=" * 70)
    print(
        f"Creating historical features for: "
        f"{player_name}"
    )
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}\n"
            "Run src/clean_data.py first."
        )

    # ========================================================
    # 1. LOAD CLEANED DATA
    # ========================================================

    df = pd.read_csv(
        input_path,
        parse_dates=["Date"],
    )

    required_columns = [
        "Match id",
        "Date",
        "Map",
        "Agent",
        "Role",
        "Kills",
        "Deaths",
        "ACS",
        "Result",
        "Win Flag",
        "Binary Target Eligible",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{player_name}: Missing required cleaned "
            f"columns: {missing_columns}"
        )

    for column in [
        "Kills",
        "Deaths",
        "ACS",
        "Win Flag",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .sort_values(
            by=["Date", "Match id"]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nLoaded {len(df)} cleaned matches."
    )

    # Internal chronological index.
    df["_match_order"] = np.arange(
        len(df)
    )

    # ========================================================
    # OVERALL PLAYER HISTORY
    # ========================================================

    df["previous_matches"] = (
        df["_match_order"]
    )

    df["previous_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .expanding()
        .mean()
    )

    previous_total_kills = (
        df["Kills"]
        .cumsum()
        .shift(1)
    )

    previous_total_deaths = (
        df["Deaths"]
        .cumsum()
        .shift(1)
    )

    df["previous_kd"] = safe_divide(
        previous_total_kills,
        previous_total_deaths,
    )

    # ========================================================
    # RECENT FORM
    # ========================================================

    df["recent_5_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    df["recent_10_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1,
        )
        .mean()
    )

    recent_5_kills = (
        df["Kills"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .sum()
    )

    recent_5_deaths = (
        df["Deaths"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .sum()
    )

    df["recent_5_kd"] = safe_divide(
        recent_5_kills,
        recent_5_deaths,
    )

    recent_10_kills = (
        df["Kills"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1,
        )
        .sum()
    )

    recent_10_deaths = (
        df["Deaths"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1,
        )
        .sum()
    )

    df["recent_10_kd"] = safe_divide(
        recent_10_kills,
        recent_10_deaths,
    )

    df["recent_5_acs"] = (
        df["ACS"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    df["recent_10_acs"] = (
        df["ACS"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1,
        )
        .mean()
    )

    # ========================================================
    # AGENT HISTORY
    # ========================================================

    df["agent_previous_matches"] = (
        df
        .groupby("Agent")
        .cumcount()
    )

    df["agent_previous_win_rate"] = (
        df
        .groupby("Agent")["Win Flag"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    agent_previous_kills = (
        df
        .groupby("Agent")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    agent_previous_deaths = (
        df
        .groupby("Agent")["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["agent_previous_kd"] = safe_divide(
        agent_previous_kills,
        agent_previous_deaths,
    )

    df["agent_previous_acs"] = (
        df
        .groupby("Agent")["ACS"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    # ========================================================
    # ROLE HISTORY
    # ========================================================

    df["role_previous_matches"] = (
        df
        .groupby("Role")
        .cumcount()
    )

    df["role_previous_win_rate"] = (
        df
        .groupby("Role")["Win Flag"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    role_previous_kills = (
        df
        .groupby("Role")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    role_previous_deaths = (
        df
        .groupby("Role")["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["role_previous_kd"] = safe_divide(
        role_previous_kills,
        role_previous_deaths,
    )

    df["role_previous_acs"] = (
        df
        .groupby("Role")["ACS"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    df["role_usage_frequency"] = safe_divide(
        df["role_previous_matches"],
        df["previous_matches"],
    )

    df["agent_role_acs_delta"] = (
        df["agent_previous_acs"]
        - df["role_previous_acs"]
    )

    # ========================================================
    # MAP HISTORY
    # ========================================================

    df["map_previous_matches"] = (
        df
        .groupby("Map")
        .cumcount()
    )

    df["map_previous_win_rate"] = (
        df
        .groupby("Map")["Win Flag"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    map_previous_kills = (
        df
        .groupby("Map")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    map_previous_deaths = (
        df
        .groupby("Map")["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["map_previous_kd"] = safe_divide(
        map_previous_kills,
        map_previous_deaths,
    )

    # ========================================================
    # AGENT x MAP HISTORY
    # ========================================================

    agent_map_group = [
        "Agent",
        "Map",
    ]

    df["agent_map_previous_matches"] = (
        df
        .groupby(agent_map_group)
        .cumcount()
    )

    df["agent_map_previous_win_rate"] = (
        df
        .groupby(agent_map_group)["Win Flag"]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .mean()
        )
    )

    agent_map_previous_kills = (
        df
        .groupby(agent_map_group)["Kills"]
        .cumsum()
        - df["Kills"]
    )

    agent_map_previous_deaths = (
        df
        .groupby(agent_map_group)["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["agent_map_previous_kd"] = safe_divide(
        agent_map_previous_kills,
        agent_map_previous_deaths,
    )

    # ========================================================
    # AGENT FAMILIARITY
    # ========================================================

    df["agent_usage_frequency"] = safe_divide(
        df["agent_previous_matches"],
        df["previous_matches"],
    )

    df["previous_agent_match_order"] = (
        df
        .groupby("Agent")["_match_order"]
        .shift(1)
    )

    df["agent_recency"] = (
        df["_match_order"]
        - df["previous_agent_match_order"]
    )

    df.loc[
        df["agent_previous_matches"] == 0,
        "agent_recency",
    ] = np.nan

    def agent_experience_level(
        matches,
    ):
        if matches == 0:
            return "New"

        if matches < 5:
            return "Very Low"

        if matches < 15:
            return "Low"

        if matches < 40:
            return "Medium"

        return "High"

    df["agent_familiarity"] = (
        df["agent_previous_matches"]
        .apply(
            agent_experience_level
        )
    )

    # ========================================================
    # FINAL FEATURE DATASET
    # ========================================================

    feature_columns = [
        # Identification
        "Match id",
        "Date",

        # Current pre-match information
        "Map",
        "Agent",
        "Role",

        # Overall historical information
        "previous_matches",
        "previous_win_rate",
        "previous_kd",

        # Recent form
        "recent_5_win_rate",
        "recent_10_win_rate",
        "recent_5_kd",
        "recent_10_kd",
        "recent_5_acs",
        "recent_10_acs",

        # Agent history
        "agent_previous_matches",
        "agent_previous_win_rate",
        "agent_previous_kd",
        "agent_previous_acs",

        # Role history
        "role_previous_matches",
        "role_previous_win_rate",
        "role_previous_kd",
        "role_previous_acs",
        "role_usage_frequency",
        "agent_role_acs_delta",

        # Map history
        "map_previous_matches",
        "map_previous_win_rate",
        "map_previous_kd",

        # Agent x Map history
        "agent_map_previous_matches",
        "agent_map_previous_win_rate",
        "agent_map_previous_kd",

        # Familiarity
        "agent_usage_frequency",
        "agent_recency",
        "agent_familiarity",

        # Target / reference only
        "Result",
        "Win Flag",
        "Binary Target Eligible",
    ]

    feature_df = df[
        feature_columns
    ].copy()

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n--- FEATURE ENGINEERING SUMMARY ---"
    )

    print(
        "Player:",
        player_name,
    )

    print(
        "Total rows:",
        len(feature_df),
    )

    print(
        "Total columns:",
        len(feature_df.columns),
    )

    print(
        "\nBinary eligible matches:",
        feature_df[
            "Binary Target Eligible"
        ].sum(),
    )

    print(
        "\nRows with no previous matches:",
        (
            feature_df["previous_matches"]
            == 0
        ).sum(),
    )

    print(
        "\nFirst-time agent usages:",
        (
            feature_df[
                "agent_previous_matches"
            ]
            == 0
        ).sum(),
    )

    print(
        "\nFirst-time role usages:",
        (
            feature_df[
                "role_previous_matches"
            ]
            == 0
        ).sum(),
    )

    print(
        "\nFirst-time Agent x Map combinations:",
        (
            feature_df[
                "agent_map_previous_matches"
            ]
            == 0
        ).sum(),
    )

    print("\nRole counts:")
    print(
        feature_df["Role"]
        .value_counts()
    )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    feature_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nFeature dataset saved to:"
        f"\n{output_path}"
    )

    return feature_df


# ============================================================
# RUN FEATURE ENGINEERING FOR EVERY CONFIGURED PLAYER
# ============================================================

if __name__ == "__main__":

    for dataset in DATASETS:

        player_name = (
            dataset["player_name"]
        )

        slug = (
            dataset["output_slug"]
        )

        input_path = (
            PROCESSED_DIR
            / (
                f"competitive_matches_"
                f"{slug}_cleaned.csv"
            )
        )

        output_path = (
            PROCESSED_DIR
            / (
                f"competitive_matches_"
                f"{slug}_features.csv"
            )
        )

        create_features(
            input_path=input_path,
            output_path=output_path,
            player_name=player_name,
        )

    print("\n" + "=" * 70)
    print(
        "Feature engineering completed for all "
        "configured datasets."
    )
    print("=" * 70)
