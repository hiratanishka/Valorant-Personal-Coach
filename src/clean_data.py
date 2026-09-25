from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# clean_data.py is located inside:
# valorant_personal_coach/src/clean_data.py
#
# Therefore parents[1] gives the main project folder.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Create the processed folder automatically if it does not exist.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLUMN RENAMING
# ============================================================

# The Excel files contain longer names for these columns.
# We shorten them so they are easier to use later.

COLUMN_RENAME_MAP = {
    "DDΔ (Average Damage Delta per Round)": "DDΔ",
    "HS% (Headshot Percentage)": "HS%",
    "ACS (Average Combat Score)": "ACS",
    "TRS (Tracker Score)": "TRS",
}


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

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    if result == "Draw":

        # Standard overtime draw such as:
        # 13-13, 14-14, 15-15, etc.
        if (
            team_score == opponent_score
            and team_score >= 13
        ):
            return "Overtime Draw"

        # Example: unusual 2-2 draw
        return "Review"

    # --------------------------------------------------------
    # REGULATION
    # --------------------------------------------------------

    # Standard regulation win/loss:
    # 13-0 through 13-11
    if (
        high_score == 13
        and low_score <= 11
    ):
        return "Regulation"

    # --------------------------------------------------------
    # OVERTIME
    # --------------------------------------------------------

    # Examples:
    # 14-12
    # 15-13
    # 16-14
    if (
        high_score >= 14
        and low_score >= 12
        and (high_score - low_score) == 2
    ):
        return "Overtime"

    # --------------------------------------------------------
    # SURRENDER
    # --------------------------------------------------------

    # A Win/Loss ending at a non-standard score is treated
    # as a surrendered / early-ended match.
    if result in ["Win", "Loss"]:
        return "Surrender"

    return "Review"


# ============================================================
# SURRENDERING TEAM
# ============================================================

def identify_surrendering_team(row):
    """
    Identify which side surrendered.

    IMPORTANT:
    We use the recorded Result rather than comparing the score.

    Result = Win
        -> Opponent Team surrendered

    Result = Loss
        -> Player Team surrendered
    """

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
    header_row
):
    """
    Clean one player's VALORANT Competitive match history.

    The same cleaning methodology is used for both players,
    but their datasets remain completely separate.
    """

    print("\n" + "=" * 60)
    print(f"Cleaning dataset for: {player_name}")
    print("=" * 60)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = pd.read_excel(
        input_path,
        sheet_name="Matches",
        header=header_row,
        usecols="A:V"
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
        "Team Score",
        "Opponent Score",
        "Result",
        "Kills",
        "Deaths",
        "Assists",
        "K/D",
        "ACS",
        "TRS"
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
        "Result"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    # Standardize:
    #
    # win  -> Win
    # LOSS -> Loss
    # draw -> Draw

    df["Result"] = (
        df["Result"]
        .str.title()
    )

    # ========================================================
    # 5. DATE CONVERSION
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # ========================================================
    # 6. CONVERT NUMERIC COLUMNS
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
        "TRS",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # ========================================================
    # 7. CHECK DUPLICATES
    # ========================================================

    duplicate_rows = df.duplicated().sum()

    duplicate_match_ids = (
        df["Match id"]
        .duplicated()
        .sum()
    )

    print("\nDuplicate rows:", duplicate_rows)

    print(
        "Duplicate Match IDs:",
        duplicate_match_ids
    )

    if duplicate_match_ids > 0:

        print(
            "\nWARNING: Duplicate Match IDs detected:"
        )

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
                    "Result"
                ]
            ]
        )

        raise ValueError(
            "Duplicate Match IDs found. "
            "Check the original Excel file."
        )

    # ========================================================
    # 8. VALIDATE RESULT VALUES
    # ========================================================

    valid_results = [
        "Win",
        "Loss",
        "Draw"
    ]

    invalid_results = df[
        ~df["Result"].isin(valid_results)
    ]

    if not invalid_results.empty:

        print(
            "\nUnexpected Result values found:"
        )

        print(
            invalid_results[
                [
                    "Match id",
                    "Result"
                ]
            ]
        )

        raise ValueError(
            "Invalid Result values found."
        )

    # ========================================================
    # 9. RECREATE WIN FLAG
    # ========================================================

    # We recreate Win Flag instead of trusting the Excel formula.
    #
    # Win  -> 1
    # Loss -> 0
    # Draw -> <NA>

    df["Win Flag"] = (
        df["Result"]
        .map({
            "Win": 1,
            "Loss": 0
        })
        .astype("Int64")
    )

    # ========================================================
    # 10. RECOMPUTE SCORE MARGIN
    # ========================================================

    df["Score Margin"] = (
        df["Team Score"]
        - df["Opponent Score"]
    )

    # ========================================================
    # 11. RECOMPUTE ROUNDS PLAYED
    # ========================================================

    df["Rounds Played"] = (
        df["Team Score"]
        + df["Opponent Score"]
    )

    # ========================================================
    # 12. RECOMPUTE K/D
    # ========================================================

    # K/D is undefined when Deaths = 0.
    #
    # Example:
    # Kills = 4
    # Deaths = 0
    #
    # We DO NOT create fake values such as:
    # infinity, 999, 0.5, etc.

    df["K/D"] = (
        df["Kills"]
        .div(
            df["Deaths"].replace(0, pd.NA)
        )
        .astype("Float64")
    )

    # ========================================================
    # 13. RECOMPUTE PER-ROUND STATISTICS
    # ========================================================

    valid_rounds = (
        df["Rounds Played"]
        .replace(0, pd.NA)
    )

    df["Kills/Round"] = (
        df["Kills"]
        / valid_rounds
    )

    df["Deaths/Round"] = (
        df["Deaths"]
        / valid_rounds
    )

    df["Assists/Round"] = (
        df["Assists"]
        / valid_rounds
    )

    # ========================================================
    # 14. DEATHLESS MATCH FLAG
    # ========================================================

    df["Deathless Match"] = (
        df["Deaths"] == 0
    ).astype(int)

    # ========================================================
    # 15. CLASSIFY MATCH END TYPE
    # ========================================================

    df["Match End Type"] = df.apply(
        lambda row: classify_match_end(
            row["Team Score"],
            row["Opponent Score"],
            row["Result"]
        ),
        axis=1
    )

    # ========================================================
    # 16. IDENTIFY SURRENDERING SIDE
    # ========================================================

    df["Surrendered By"] = df.apply(
        identify_surrendering_team,
        axis=1
    )

    # ========================================================
    # 17. BINARY MODEL ELIGIBILITY
    # ========================================================

    # Win/Loss matches:
    # usable for supporting Win/Loss model
    #
    # Draw:
    # retained for EDA but excluded from binary model

    df["Binary Target Eligible"] = (
        df["Result"]
        .isin(
            [
                "Win",
                "Loss"
            ]
        )
        .astype(int)
    )

    # ========================================================
    # 18. SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df
        .sort_values(
            by=[
                "Date",
                "Match id"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 19. DATA QUALITY CHECKS
    # ========================================================

    print("\n--- DATA QUALITY CHECK ---")

    print(
        "Missing Dates:",
        df["Date"].isna().sum()
    )

    print(
        "Missing Maps:",
        df["Map"].isna().sum()
    )

    print(
        "Missing Agents:",
        df["Agent"].isna().sum()
    )

    print(
        "Missing Results:",
        df["Result"].isna().sum()
    )

    print(
        "Missing K/D:",
        df["K/D"].isna().sum()
    )

    print(
        "Missing Win Flag:",
        df["Win Flag"].isna().sum()
    )

    # ========================================================
    # 20. CLEANING SUMMARY
    # ========================================================

    print("\n--- CLEANING SUMMARY ---")

    print(
        f"\nPlayer: {player_name}"
    )

    print(
        "Total matches:",
        len(df)
    )

    print("\nResults:")

    print(
        df["Result"]
        .value_counts()
    )

    print("\nMatch End Types:")

    print(
        df["Match End Type"]
        .value_counts()
    )

    print("\nSurrendered By:")

    print(
        df["Surrendered By"]
        .value_counts(
            dropna=True
        )
    )

    print(
        "\nBinary-model eligible matches:",
        df["Binary Target Eligible"].sum()
    )

    # ========================================================
    # 21. DISPLAY MATCHES REQUIRING REVIEW
    # ========================================================

    review_matches = df[
        df["Match End Type"] == "Review"
    ]

    if not review_matches.empty:

        print(
            "\nMatches requiring manual review:"
        )

        print(
            review_matches[
                [
                    "Match id",
                    "Date",
                    "Map",
                    "Agent",
                    "Team Score",
                    "Opponent Score",
                    "Result"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo matches require manual review."
        )

    # ========================================================
    # 22. SAVE CLEAN DATASET
    # ========================================================

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCleaned dataset saved to:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 60)

    return df


# ============================================================
# RUN CLEANING FOR BOTH PLAYERS
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # TANISHKA
    # ========================================================

    clean_dataset(
        input_path=(
            DATA_DIR
            / "competitive_matches.xlsx"
        ),

        output_path=(
            PROCESSED_DIR
            / "competitive_matches_cleaned.csv"
        ),

        player_name="Tanishka",

        # Your Excel:
        # row 1 = title
        # row 2 = headers
        header_row=1
    )

    # ========================================================
    # DEV
    # ========================================================

    clean_dataset(
        input_path=(
            DATA_DIR
            / "competitive_matches_dev.xlsx"
        ),

        output_path=(
            PROCESSED_DIR
            / "competitive_matches_dev_cleaned.csv"
        ),

        player_name="Dev",

        # Dev Excel:
        # row 1 = title
        # row 2 = blank
        # row 3 = headers
        header_row=2
    )