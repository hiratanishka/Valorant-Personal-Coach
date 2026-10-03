from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
UPDATED_DIR = DATA_DIR / "updated"
UPDATED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLUMN RENAMING
# ============================================================

COLUMN_RENAME_MAP = {
    "DDΔ (Average Damage Delta per Round)": "DDΔ",
    "HS% (Headshot Percentage)": "HS%",
    "ACS (Average Combat Score)": "ACS",
}


# ============================================================
# AGENT -> ROLE MAPPING
# ============================================================

AGENT_ROLE_MAP = {
    # Duelists
    "Iso": "Duelist",
    "Jett": "Duelist",
    "Neon": "Duelist",
    "Phoenix": "Duelist",
    "Raze": "Duelist",
    "Reyna": "Duelist",
    "Waylay": "Duelist",
    "Yoru": "Duelist",

    # Controllers
    "Astra": "Controller",
    "Brimstone": "Controller",
    "Clove": "Controller",
    "Harbor": "Controller",
    "Miks": "Controller",
    "Omen": "Controller",
    "Viper": "Controller",

    # Initiators
    "Breach": "Initiator",
    "Fade": "Initiator",
    "Gekko": "Initiator",
    "KAY/O": "Initiator",
    "Skye": "Initiator",
    "Sova": "Initiator",
    "Tejo": "Initiator",

    # Sentinels
    "Chamber": "Sentinel",
    "Cypher": "Sentinel",
    "Deadlock": "Sentinel",
    "Killjoy": "Sentinel",
    "Sage": "Sentinel",
    "Veto": "Sentinel",
    "Vyse": "Sentinel",
}

VALID_ROLES = {
    "Duelist",
    "Controller",
    "Initiator",
    "Sentinel",
}


def expected_role(agent):
    """Return the official role expected for an agent."""
    return AGENT_ROLE_MAP.get(agent)


# ============================================================
# MATCH END TYPE
# ============================================================

def classify_match_end(team_score, opponent_score, result):
    """
    Classify how a VALORANT Competitive match ended.

    Possible values:
    - Regulation
    - Overtime
    - Overtime Draw
    - Surrender
    - Review
    """

    high_score = max(team_score, opponent_score)
    low_score = min(team_score, opponent_score)

    if result == "Draw":
        if (
            team_score == opponent_score
            and team_score >= 13
        ):
            return "Overtime Draw"

        return "Review"

    if (
        high_score == 13
        and low_score <= 11
    ):
        return "Regulation"

    if (
        high_score >= 14
        and low_score >= 12
        and (high_score - low_score) == 2
    ):
        return "Overtime"

    if result in ["Win", "Loss"]:
        return "Surrender"

    return "Review"


# ============================================================
# SURRENDERING TEAM
# ============================================================

def identify_surrendering_team(row):
    """Identify which side surrendered using the recorded result."""

    if row["Match End Type"] != "Surrender":
        return pd.NA

    if row["Result"] == "Win":
        return "Opponent Team"

    if row["Result"] == "Loss":
        return "Player Team"

    return pd.NA


# ============================================================
# CLEAN ONE PLAYER DATASET
# ============================================================

def clean_dataset(
    input_path,
    output_path,
    player_name,
    header_row,
):
    """
    Clean one player's updated VALORANT Competitive match history.

    Important project rules:
    - Tanishka and Dev remain separate datasets.
    - TRS is no longer used.
    - Role is a required categorical variable.
    - The agent determines the official Role.
    - ACS is retained as a performance statistic, but current-match ACS
      is not used directly as a pre-match model feature later.
    """

    print("\n" + "=" * 65)
    print(f"Cleaning updated dataset for: {player_name}")
    print("=" * 65)

    # ========================================================
    # 1. LOAD UPDATED DATA
    # ========================================================

    df = pd.read_excel(
        input_path,
        sheet_name="Matches",
        header=header_row,
        usecols="A:V",
    )

    print(f"\nLoaded {len(df)} matches.")

    # ========================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = df.rename(
        columns=COLUMN_RENAME_MAP
    )

    required_columns = [
        "Match id",
        "Date",
        "Map",
        "Agent",
        "Role",
        "Team Score",
        "Opponent Score",
        "Result",
        "Kills",
        "Deaths",
        "Assists",
        "K/D",
        "ACS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if "TRS" in df.columns:
        raise ValueError(
            "TRS is still present in the updated dataset. "
            "Use the revised Role-aware Excel file."
        )

    print("\nColumns:")
    print(df.columns.tolist())

    # ========================================================
    # 3. REMOVE COMPLETELY EMPTY ROWS
    # ========================================================

    df = (
        df
        .dropna(how="all")
        .copy()
    )

    # ========================================================
    # 4. CLEAN TEXT COLUMNS
    # ========================================================

    text_columns = [
        "Map",
        "Placement",
        "Agent",
        "Role",
        "Result",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    df["Result"] = (
        df["Result"]
        .str.title()
    )

    # ========================================================
    # 5. VALIDATE AGENT -> ROLE
    # ========================================================

    unknown_agents = sorted(
        df.loc[
            ~df["Agent"].isin(AGENT_ROLE_MAP),
            "Agent",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_agents:
        raise ValueError(
            "Agents missing from AGENT_ROLE_MAP: "
            f"{unknown_agents}"
        )

    invalid_roles = sorted(
        df.loc[
            ~df["Role"].isin(VALID_ROLES),
            "Role",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if invalid_roles:
        raise ValueError(
            f"Invalid Role values found: {invalid_roles}"
        )

    expected_roles = df["Agent"].map(AGENT_ROLE_MAP)

    role_mismatches = df[
        df["Role"] != expected_roles
    ]

    if not role_mismatches.empty:
        print("\nAgent/Role mismatches found:")
        print(
            role_mismatches[
                ["Match id", "Agent", "Role"]
            ].to_string(index=False)
        )

        raise ValueError(
            "Role does not match the official Agent -> Role mapping."
        )

    # Reassign from the official map after validation so downstream
    # files always use one consistent source of truth.
    df["Role"] = expected_roles.astype("string")

    # ========================================================
    # 6. DATE CONVERSION
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    # ========================================================
    # 7. CONVERT NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [
        "Match id",
        "Team Score",
        "Opponent Score",
        "Kills",
        "Deaths",
        "Assists",
        "K/D",
        "Kills/Round",
        "Deaths/Round",
        "Assists/Round",
        "DDΔ",
        "HS%",
        "ACS",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ========================================================
    # 8. CHECK DUPLICATES
    # ========================================================

    duplicate_rows = df.duplicated().sum()

    duplicate_match_ids = (
        df["Match id"]
        .duplicated()
        .sum()
    )

    print("\nDuplicate rows:", duplicate_rows)
    print("Duplicate Match IDs:", duplicate_match_ids)

    if duplicate_match_ids > 0:
        print("\nWARNING: Duplicate Match IDs detected:")

        print(
            df[
                df["Match id"].duplicated(
                    keep=False
                )
            ][
                [
                    "Match id",
                    "Date",
                    "Map",
                    "Agent",
                    "Role",
                    "Result",
                ]
            ]
        )

        raise ValueError(
            "Duplicate Match IDs found. Check the original Excel file."
        )

    # ========================================================
    # 9. VALIDATE RESULT VALUES
    # ========================================================

    valid_results = [
        "Win",
        "Loss",
        "Draw",
    ]

    invalid_results = df[
        ~df["Result"].isin(valid_results)
    ]

    if not invalid_results.empty:
        print("\nUnexpected Result values found:")
        print(
            invalid_results[
                ["Match id", "Result"]
            ]
        )

        raise ValueError(
            "Invalid Result values found."
        )

    # ========================================================
    # 10. RECREATE WIN FLAG
    # ========================================================

    df["Win Flag"] = (
        df["Result"]
        .map({
            "Win": 1,
            "Loss": 0,
        })
        .astype("Int64")
    )

    # ========================================================
    # 11. RECOMPUTE SCORE MARGIN
    # ========================================================

    df["Score Margin"] = (
        df["Team Score"]
        - df["Opponent Score"]
    )

    # ========================================================
    # 12. RECOMPUTE ROUNDS PLAYED
    # ========================================================

    df["Rounds Played"] = (
        df["Team Score"]
        + df["Opponent Score"]
    )

    # ========================================================
    # 13. RECOMPUTE K/D
    # ========================================================

    df["K/D"] = (
        df["Kills"]
        .div(
            df["Deaths"].replace(0, pd.NA)
        )
        .astype("Float64")
    )

    # ========================================================
    # 14. RECOMPUTE PER-ROUND STATISTICS
    # ========================================================

    valid_rounds = (
        df["Rounds Played"]
        .replace(0, pd.NA)
    )

    df["Kills/Round"] = (
        df["Kills"] / valid_rounds
    )

    df["Deaths/Round"] = (
        df["Deaths"] / valid_rounds
    )

    df["Assists/Round"] = (
        df["Assists"] / valid_rounds
    )

    # ========================================================
    # 15. DEATHLESS MATCH FLAG
    # ========================================================

    df["Deathless Match"] = (
        df["Deaths"] == 0
    ).astype(int)

    # ========================================================
    # 16. CLASSIFY MATCH END TYPE
    # ========================================================

    df["Match End Type"] = df.apply(
        lambda row: classify_match_end(
            row["Team Score"],
            row["Opponent Score"],
            row["Result"],
        ),
        axis=1,
    )

    # ========================================================
    # 17. IDENTIFY SURRENDERING SIDE
    # ========================================================

    df["Surrendered By"] = df.apply(
        identify_surrendering_team,
        axis=1,
    )

    # ========================================================
    # 18. BINARY MODEL ELIGIBILITY
    # ========================================================

    df["Binary Target Eligible"] = (
        df["Result"]
        .isin(["Win", "Loss"])
        .astype(int)
    )

    # ========================================================
    # 19. SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df
        .sort_values(
            by=["Date", "Match id"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 20. DATA QUALITY CHECKS
    # ========================================================

    print("\n--- DATA QUALITY CHECK ---")

    print("Missing Dates:", df["Date"].isna().sum())
    print("Missing Maps:", df["Map"].isna().sum())
    print("Missing Agents:", df["Agent"].isna().sum())
    print("Missing Roles:", df["Role"].isna().sum())
    print("Missing Results:", df["Result"].isna().sum())
    print("Missing K/D:", df["K/D"].isna().sum())
    print("Missing ACS:", df["ACS"].isna().sum())
    print("Missing Win Flag:", df["Win Flag"].isna().sum())

    # ========================================================
    # 21. CLEANING SUMMARY
    # ========================================================

    print("\n--- CLEANING SUMMARY ---")
    print(f"\nPlayer: {player_name}")
    print("Total matches:", len(df))

    print("\nResults:")
    print(df["Result"].value_counts())

    print("\nRole distribution:")
    print(df["Role"].value_counts())

    print("\nMatch End Types:")
    print(df["Match End Type"].value_counts())

    print("\nSurrendered By:")
    print(
        df["Surrendered By"]
        .value_counts(dropna=True)
    )

    print(
        "\nBinary-model eligible matches:",
        df["Binary Target Eligible"].sum(),
    )

    # ========================================================
    # 22. DISPLAY MATCHES REQUIRING REVIEW
    # ========================================================

    review_matches = df[
        df["Match End Type"] == "Review"
    ]

    if not review_matches.empty:
        print("\nMatches requiring manual review:")
        print(
            review_matches[
                [
                    "Match id",
                    "Date",
                    "Map",
                    "Agent",
                    "Role",
                    "Team Score",
                    "Opponent Score",
                    "Result",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nNo matches require manual review.")

    # ========================================================
    # 23. SAVE CLEAN DATASET
    # ========================================================

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nCleaned dataset saved to:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 65)

    return df


# ============================================================
# RUN CLEANING FOR BOTH PLAYERS
# ============================================================

if __name__ == "__main__":

    # Tanishka updated dataset
    clean_dataset(
        input_path=(
            UPDATED_DIR
            / "competitive_matches_role_updated.xlsx"
        ),
        output_path=(
            UPDATED_DIR
            / "competitive_matches_cleaned.csv"
        ),
        player_name="Tanishka",
        header_row=1,
    )

    # Dev updated dataset
    clean_dataset(
        input_path=(
            UPDATED_DIR
            / "competitive_matches_dev_role_updated.xlsx"
        ),
        output_path=(
            UPDATED_DIR
            / "competitive_matches_dev_cleaned.csv"
        ),
        player_name="Dev",
        header_row=2,
    )
