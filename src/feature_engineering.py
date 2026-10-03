from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
UPDATED_DIR = DATA_DIR / "updated"

UPDATED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator):
    """Divide two pandas Series while avoiding division by zero."""

    denominator = denominator.replace(0, np.nan)
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

    Every feature for Match N is calculated using only matches
    that happened BEFORE Match N.

    Role is known before the match because it is determined by
    the selected agent. ACS is never taken from the current match;
    only historical ACS summaries are used.
    """

    print("\n" + "=" * 65)
    print(f"Creating historical features for: {player_name}")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD CLEANED DATA
    # --------------------------------------------------------

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
            f"Missing required cleaned columns: {missing_columns}"
        )

    # Ensure numeric columns are numeric after CSV import.
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

    # Always preserve chronological order.
    df = (
        df
        .sort_values(
            by=["Date", "Match id"]
        )
        .reset_index(drop=True)
    )

    print(f"\nLoaded {len(df)} cleaned matches.")

    # --------------------------------------------------------
    # 2. INTERNAL CHRONOLOGICAL INDEX
    # --------------------------------------------------------

    df["_match_order"] = np.arange(len(df))

    # ========================================================
    # OVERALL PLAYER HISTORY
    # ========================================================

    df["previous_matches"] = df["_match_order"]

    # --------------------------------------------------------
    # 3. OVERALL PREVIOUS WIN RATE
    # --------------------------------------------------------

    df["previous_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .expanding()
        .mean()
    )

    # --------------------------------------------------------
    # 4. OVERALL PREVIOUS K/D
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. RECENT 5-MATCH WIN RATE
    # --------------------------------------------------------

    df["recent_5_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # 6. RECENT 10-MATCH WIN RATE
    # --------------------------------------------------------

    df["recent_10_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # 7. RECENT 5-MATCH K/D
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. RECENT 10-MATCH K/D
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 9. RECENT 5-MATCH ACS
    # --------------------------------------------------------

    df["recent_5_acs"] = (
        df["ACS"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # 10. RECENT 10-MATCH ACS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 11. PREVIOUS MATCHES WITH CURRENT AGENT
    # --------------------------------------------------------

    df["agent_previous_matches"] = (
        df
        .groupby("Agent")
        .cumcount()
    )

    # --------------------------------------------------------
    # 12. PREVIOUS AGENT WIN RATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 13. PREVIOUS AGENT K/D
    # --------------------------------------------------------

    agent_previous_kills = (
        df.groupby("Agent")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    agent_previous_deaths = (
        df.groupby("Agent")["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["agent_previous_kd"] = safe_divide(
        agent_previous_kills,
        agent_previous_deaths,
    )

    # --------------------------------------------------------
    # 14. PREVIOUS AGENT ACS
    # --------------------------------------------------------

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

    # The current Role itself is pre-match information because
    # the selected agent determines the official role.

    # --------------------------------------------------------
    # 15. PREVIOUS MATCHES WITH CURRENT ROLE
    # --------------------------------------------------------

    df["role_previous_matches"] = (
        df
        .groupby("Role")
        .cumcount()
    )

    # --------------------------------------------------------
    # 16. PREVIOUS ROLE WIN RATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 17. PREVIOUS ROLE K/D
    # --------------------------------------------------------

    role_previous_kills = (
        df.groupby("Role")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    role_previous_deaths = (
        df.groupby("Role")["Deaths"]
        .cumsum()
        - df["Deaths"]
    )

    df["role_previous_kd"] = safe_divide(
        role_previous_kills,
        role_previous_deaths,
    )

    # --------------------------------------------------------
    # 18. PREVIOUS ROLE ACS
    # --------------------------------------------------------

    # IMPORTANT:
    # This is historical ACS for the role only. The current
    # match ACS is shifted out and never used as input.

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

    # --------------------------------------------------------
    # 19. ROLE USAGE FREQUENCY
    # --------------------------------------------------------

    df["role_usage_frequency"] = safe_divide(
        df["role_previous_matches"],
        df["previous_matches"],
    )

    # --------------------------------------------------------
    # 20. AGENT ACS RELATIVE TO ROLE ACS
    # --------------------------------------------------------

    # Positive value:
    # the player's historical ACS with this agent is above
    # their historical ACS baseline for the role.
    #
    # Negative value:
    # the agent's historical ACS is below that role baseline.

    df["agent_role_acs_delta"] = (
        df["agent_previous_acs"]
        - df["role_previous_acs"]
    )

    # ========================================================
    # MAP HISTORY
    # ========================================================

    # --------------------------------------------------------
    # 21. PREVIOUS MATCHES ON CURRENT MAP
    # --------------------------------------------------------

    df["map_previous_matches"] = (
        df
        .groupby("Map")
        .cumcount()
    )

    # --------------------------------------------------------
    # 22. PREVIOUS MAP WIN RATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 23. PREVIOUS MAP K/D
    # --------------------------------------------------------

    map_previous_kills = (
        df.groupby("Map")["Kills"]
        .cumsum()
        - df["Kills"]
    )

    map_previous_deaths = (
        df.groupby("Map")["Deaths"]
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

    # --------------------------------------------------------
    # 24. PREVIOUS AGENT x MAP MATCHES
    # --------------------------------------------------------

    df["agent_map_previous_matches"] = (
        df
        .groupby(agent_map_group)
        .cumcount()
    )

    # --------------------------------------------------------
    # 25. PREVIOUS AGENT x MAP WIN RATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 26. PREVIOUS AGENT x MAP K/D
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 27. AGENT USAGE FREQUENCY
    # --------------------------------------------------------

    df["agent_usage_frequency"] = safe_divide(
        df["agent_previous_matches"],
        df["previous_matches"],
    )

    # --------------------------------------------------------
    # 28. AGENT RECENCY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 29. AGENT EXPERIENCE LEVEL
    # --------------------------------------------------------

    def agent_experience_level(matches):
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
        .apply(agent_experience_level)
    )

    # ========================================================
    # TARGET
    # ========================================================

    # Win Flag is the supporting binary target:
    # Win = 1, Loss = 0, Draw = NaN.
    # It is NOT an input feature.

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

        # Target / reference
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

    print("\n--- FEATURE ENGINEERING SUMMARY ---")

    print("Total rows:", len(feature_df))
    print("Total columns:", len(feature_df.columns))

    print(
        "\nBinary eligible matches:",
        feature_df["Binary Target Eligible"].sum(),
    )

    print(
        "\nRows with no previous matches:",
        (feature_df["previous_matches"] == 0).sum(),
    )

    print(
        "\nFirst-time agent usages:",
        (feature_df["agent_previous_matches"] == 0).sum(),
    )

    print(
        "\nFirst-time role usages:",
        (feature_df["role_previous_matches"] == 0).sum(),
    )

    print(
        "\nFirst-time Agent x Map combinations:",
        (feature_df["agent_map_previous_matches"] == 0).sum(),
    )

    print("\nRole counts:")
    print(feature_df["Role"].value_counts())

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

    print("\n" + "=" * 65)

    return feature_df


# ============================================================
# RUN FOR BOTH PLAYERS
# ============================================================

if __name__ == "__main__":

    # Tanishka
    create_features(
        input_path=(
            UPDATED_DIR
            / "competitive_matches_cleaned.csv"
        ),
        output_path=(
            UPDATED_DIR
            / "competitive_matches_features.csv"
        ),
        player_name="Tanishka",
    )

    # Dev
    create_features(
        input_path=(
            UPDATED_DIR
            / "competitive_matches_dev_cleaned.csv"
        ),
        output_path=(
            UPDATED_DIR
            / "competitive_matches_dev_features.csv"
        ),
        player_name="Dev",
    )
