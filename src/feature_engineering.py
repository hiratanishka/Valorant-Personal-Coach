from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator):
    """
    Divide two pandas Series while avoiding division by zero.
    """

    denominator = denominator.replace(0, np.nan)

    return numerator / denominator


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    input_path,
    output_path,
    player_name
):
    """
    Create leakage-free historical features for one player's
    VALORANT Competitive match history.

    Every feature for Match N is calculated using only matches
    that happened BEFORE Match N.
    """

    print("\n" + "=" * 65)
    print(f"Creating historical features for: {player_name}")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD CLEANED DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path,
        parse_dates=["Date"]
    )

    # Always make sure chronological order is preserved.
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

    # Used only for calculating previous history / recency.
    df["_match_order"] = np.arange(len(df))

    # ========================================================
    # OVERALL PLAYER HISTORY
    # ========================================================

    # Number of matches played BEFORE current match.
    df["previous_matches"] = df["_match_order"]

    # --------------------------------------------------------
    # 3. OVERALL PREVIOUS WIN RATE
    # --------------------------------------------------------

    # shift(1) is critical:
    # current match result is NEVER included.

    df["previous_win_rate"] = (
        df["Win Flag"]
        .shift(1)
        .expanding()
        .mean()
    )

    # --------------------------------------------------------
    # 4. OVERALL PREVIOUS K/D
    # --------------------------------------------------------

    # We use:
    #
    # previous total kills / previous total deaths
    #
    # rather than averaging individual-match K/D values.

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
        previous_total_deaths
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
            min_periods=1
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
            min_periods=1
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
            min_periods=1
        )
        .sum()
    )

    recent_5_deaths = (
        df["Deaths"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1
        )
        .sum()
    )

    df["recent_5_kd"] = safe_divide(
        recent_5_kills,
        recent_5_deaths
    )

    # --------------------------------------------------------
    # 8. RECENT 10-MATCH K/D
    # --------------------------------------------------------

    recent_10_kills = (
        df["Kills"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1
        )
        .sum()
    )

    recent_10_deaths = (
        df["Deaths"]
        .shift(1)
        .rolling(
            window=10,
            min_periods=1
        )
        .sum()
    )

    df["recent_10_kd"] = safe_divide(
        recent_10_kills,
        recent_10_deaths
    )

    # --------------------------------------------------------
    # 9. RECENT 5-MATCH ACS
    # --------------------------------------------------------

    df["recent_5_acs"] = (
        df["ACS"]
        .shift(1)
        .rolling(
            window=5,
            min_periods=1
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
            min_periods=1
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
        agent_previous_deaths
    )

    # --------------------------------------------------------
    # 14. PREVIOUS AGENT ACS
    # --------------------------------------------------------

    agent_previous_acs_sum = (
        df.groupby("Agent")["ACS"]
        .cumsum()
        - df["ACS"]
    )

    df["agent_previous_acs"] = safe_divide(
        agent_previous_acs_sum,
        df["agent_previous_matches"]
    )

    # ========================================================
    # MAP HISTORY
    # ========================================================

    # --------------------------------------------------------
    # 15. PREVIOUS MATCHES ON CURRENT MAP
    # --------------------------------------------------------

    df["map_previous_matches"] = (
        df
        .groupby("Map")
        .cumcount()
    )

    # --------------------------------------------------------
    # 16. PREVIOUS MAP WIN RATE
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
    # 17. PREVIOUS MAP K/D
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
        map_previous_deaths
    )

    # ========================================================
    # AGENT × MAP HISTORY
    # ========================================================

    agent_map_group = [
        "Agent",
        "Map"
    ]

    # --------------------------------------------------------
    # 18. PREVIOUS AGENT × MAP MATCHES
    # --------------------------------------------------------

    df["agent_map_previous_matches"] = (
        df
        .groupby(agent_map_group)
        .cumcount()
    )

    # --------------------------------------------------------
    # 19. PREVIOUS AGENT × MAP WIN RATE
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
    # 20. PREVIOUS AGENT × MAP K/D
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
        agent_map_previous_deaths
    )

    # ========================================================
    # AGENT FAMILIARITY
    # ========================================================

    # --------------------------------------------------------
    # 21. AGENT USAGE FREQUENCY
    # --------------------------------------------------------

    # Example:
    #
    # Before current match:
    #
    # Previous total matches = 100
    # Previous Sage matches = 40
    #
    # Agent Usage Frequency = 0.40

    df["agent_usage_frequency"] = safe_divide(
        df["agent_previous_matches"],
        df["previous_matches"]
    )

    # --------------------------------------------------------
    # 22. AGENT RECENCY
    # --------------------------------------------------------

    # Find the global match position where this agent
    # was previously played.

    df["previous_agent_match_order"] = (
        df
        .groupby("Agent")["_match_order"]
        .shift(1)
    )

    # Number of matches since the agent was last used.
    df["agent_recency"] = (
        df["_match_order"]
        - df["previous_agent_match_order"]
    )

    # First time using an agent -> no previous usage.
    df.loc[
        df["agent_previous_matches"] == 0,
        "agent_recency"
    ] = np.nan

    # ========================================================
    # EXPERIENCE / FAMILIARITY FLAGS
    # ========================================================

    # --------------------------------------------------------
    # 23. AGENT EXPERIENCE LEVEL
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

    # Win Flag remains our supporting binary target:
    #
    # Win  = 1
    # Loss = 0
    # Draw = NaN
    #
    # IMPORTANT:
    # Win Flag is NOT an input feature.
    # It is the value the model learns to estimate.

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

        # Map history
        "map_previous_matches",
        "map_previous_win_rate",
        "map_previous_kd",

        # Agent × Map history
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
        "Binary Target Eligible"
    ]

    feature_df = df[
        feature_columns
    ].copy()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n--- FEATURE ENGINEERING SUMMARY ---")

    print(
        "Total rows:",
        len(feature_df)
    )

    print(
        "Total columns:",
        len(feature_df.columns)
    )

    print(
        "\nBinary eligible matches:",
        feature_df[
            "Binary Target Eligible"
        ].sum()
    )

    print(
        "\nRows with no previous matches:",
        (
            feature_df["previous_matches"] == 0
        ).sum()
    )

    print(
        "\nFirst-time agent usages:",
        (
            feature_df[
                "agent_previous_matches"
            ] == 0
        ).sum()
    )

    print(
        "\nFirst-time Agent × Map combinations:",
        (
            feature_df[
                "agent_map_previous_matches"
            ] == 0
        ).sum()
    )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    feature_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
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

    # --------------------------------------------------------
    # TANISHKA
    # --------------------------------------------------------

    create_features(
        input_path=(
            PROCESSED_DIR
            / "competitive_matches_cleaned.csv"
        ),

        output_path=(
            PROCESSED_DIR
            / "competitive_matches_features.csv"
        ),

        player_name="Tanishka"
    )

    # --------------------------------------------------------
    # DEV
    # --------------------------------------------------------

    create_features(
        input_path=(
            PROCESSED_DIR
            / "competitive_matches_dev_cleaned.csv"
        ),

        output_path=(
            PROCESSED_DIR
            / "competitive_matches_dev_features.csv"
        ),

        player_name="Dev"
    )