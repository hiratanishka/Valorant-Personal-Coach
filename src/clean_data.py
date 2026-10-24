from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

# Raw Excel files stay directly inside data/
RAW_DATA_DIR = DATA_DIR

# Generated cleaned + feature CSV files go here.
PROCESSED_DIR = DATA_DIR / "datasets"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Add Player 4 and Player 5 here later.
#
# Example:
# {
#     "player_name": "Player4",
#     "input_filename": "competitive_matches_player4.xlsx",
#     "output_slug": "player4",
# },

DATASETS = [
    {
        "player_name": "Tanishka",
        "input_filename": "competitive_matches.xlsx",
        "output_slug": "tanishka",
    },
    {
        "player_name": "Dev",
        "input_filename": "competitive_matches_dev.xlsx",
        "output_slug": "dev",
    },
    {
        "player_name": "Anuj",
        "input_filename": "competitive_matches_anuj.xlsx",
        "output_slug": "anuj",
    },
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================
# The three current Excel files do not use exactly the same
# header wording. They are normalized here into one schema.

COLUMN_RENAME_MAP = {
    "Match ID": "Match id",
    "DDΔ (Per Round)": "DDΔ",
    "DDΔ (Average Damage Delta per Round)": "DDΔ",
    "HS% (Headshot Percentage)": "HS%",
    "ACS (Average Combat Score)": "ACS",
    "TRS (Tracker Score)": "TRS",
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


# ============================================================
# HEADER AUTO-DETECTION
# ============================================================

def detect_header_row(input_path):
    """
    Detect the row containing the real Matches-table header.

    This lets the same cleaner handle files where the header
    begins on row 2, row 3, row 4, etc.
    """

    preview = pd.read_excel(
        input_path,
        sheet_name="Matches",
        header=None,
        nrows=12,
    )

    for row_number, row in preview.iterrows():
        values = {
            str(value).strip().lower()
            for value in row.dropna().tolist()
        }

        has_match_id = "match id" in values
        has_date = "date" in values
        has_agent = "agent" in values

        if has_match_id and has_date and has_agent:
            return int(row_number)

    raise ValueError(
        f"Could not detect the Matches header row in: {input_path}"
    )


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

    if pd.isna(team_score) or pd.isna(opponent_score):
        return "Review"

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
):
    """
    Clean one player's raw VALORANT Competitive match history.

    Final project rules:
    - Every player remains a separate dataset.
    - TRS is removed from the final ML pipeline.
    - Role is generated from Agent identity.
    - ACS is retained as a performance statistic.
    - Current-match ACS is not used directly later as a
      pre-match model feature.
    """

    print("\n" + "=" * 70)
    print(f"Cleaning dataset for: {player_name}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    header_row = detect_header_row(input_path)

    print(
        f"\nDetected Matches header row: "
        f"Excel row {header_row + 1}"
    )

    # ========================================================
    # 1. LOAD RAW DATA
    # ========================================================

    df = pd.read_excel(
        input_path,
        sheet_name="Matches",
        header=header_row,
    )

    print(f"Loaded {len(df)} rows.")

    # ========================================================
    # 2. NORMALIZE COLUMN NAMES
    # ========================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = df.rename(
        columns=COLUMN_RENAME_MAP
    )

    # Drop completely unnamed/helper columns.
    helper_columns = [
        column
        for column in df.columns
        if (
            str(column).startswith("Unnamed:")
            or str(column).startswith("http://")
            or str(column).startswith("https://")
        )
    ]

    df = df.drop(
        columns=helper_columns,
        errors="ignore",
    )

    # TRS belongs to the original raw EDA only and is intentionally
    # excluded from the revised modelling pipeline.
    trs_columns = [
        column
        for column in df.columns
        if str(column).strip().lower().startswith("trs")
    ]

    if trs_columns:
        print(
            "\nRemoving TRS from final modelling data:",
            trs_columns,
        )

        df = df.drop(
            columns=trs_columns,
            errors="ignore",
        )

    # ========================================================
    # 3. VALIDATE REQUIRED RAW COLUMNS
    # ========================================================

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
        "ACS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{player_name}: Missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # 4. REMOVE EMPTY ROWS
    # ========================================================

    df = (
        df
        .dropna(how="all")
        .copy()
    )

    # ========================================================
    # 5. CLEAN TEXT COLUMNS
    # ========================================================

    text_columns = [
        "Map",
        "Placement",
        "Agent",
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
    # 6. ADD OFFICIAL ROLE FROM AGENT
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
            f"{player_name}: Agents missing from "
            f"AGENT_ROLE_MAP: {unknown_agents}"
        )

    df["Role"] = (
        df["Agent"]
        .map(AGENT_ROLE_MAP)
        .astype("string")
    )

    # Place Role immediately after Agent.
    role_series = df.pop("Role")
    agent_position = (
        df.columns
        .get_loc("Agent")
    )

    df.insert(
        agent_position + 1,
        "Role",
        role_series,
    )

    # ========================================================
    # 7. DATE CONVERSION
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    # ========================================================
    # 8. NUMERIC CONVERSION
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
    # 9. DUPLICATE CHECK
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
        raise ValueError(
            f"{player_name}: Duplicate Match IDs found. "
            "Check the raw Excel file."
        )

    # ========================================================
    # 10. VALIDATE RESULT VALUES
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
        raise ValueError(
            f"{player_name}: Invalid Result values found: "
            f"{sorted(invalid_results['Result'].dropna().unique())}"
        )

    # ========================================================
    # 11. RECREATE DERIVED MATCH VALUES
    # ========================================================

    df["Win Flag"] = (
        df["Result"]
        .map({
            "Win": 1,
            "Loss": 0,
        })
        .astype("Int64")
    )

    df["Score Margin"] = (
        df["Team Score"]
        - df["Opponent Score"]
    )

    df["Rounds Played"] = (
        df["Team Score"]
        + df["Opponent Score"]
    )

    df["K/D"] = (
        df["Kills"]
        .div(
            df["Deaths"].replace(0, pd.NA)
        )
        .astype("Float64")
    )

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

    df["Deathless Match"] = (
        df["Deaths"] == 0
    ).astype(int)

    # ========================================================
    # 12. MATCH END INFORMATION
    # ========================================================

    df["Match End Type"] = df.apply(
        lambda row: classify_match_end(
            row["Team Score"],
            row["Opponent Score"],
            row["Result"],
        ),
        axis=1,
    )

    df["Surrendered By"] = df.apply(
        identify_surrendering_team,
        axis=1,
    )

    # ========================================================
    # 13. BINARY MODEL ELIGIBILITY
    # ========================================================

    df["Binary Target Eligible"] = (
        df["Result"]
        .isin(["Win", "Loss"])
        .astype(int)
    )

    # ========================================================
    # 14. SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df
        .sort_values(
            by=["Date", "Match id"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 15. FINAL CLEANED COLUMN ORDER
    # ========================================================

    final_columns = [
        "Match id",
        "Date",
        "Map",
        "Placement",
        "Agent",
        "Role",
        "Team Score",
        "Opponent Score",
        "Result",
        "Win Flag",
        "Score Margin",
        "Rounds Played",
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
        "Deathless Match",
        "Match End Type",
        "Surrendered By",
        "Binary Target Eligible",
    ]

    final_columns = [
        column
        for column in final_columns
        if column in df.columns
    ]

    df = df[
        final_columns
    ].copy()

    # ========================================================
    # 16. DATA QUALITY SUMMARY
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

    print("\n--- CLEANING SUMMARY ---")
    print("Player:", player_name)
    print("Total matches:", len(df))

    print("\nResults:")
    print(df["Result"].value_counts())

    print("\nRole distribution:")
    print(df["Role"].value_counts())

    print(
        "\nBinary-model eligible matches:",
        df["Binary Target Eligible"].sum(),
    )

    # ========================================================
    # 17. SAVE
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

    return df


# ============================================================
# RUN CLEANING FOR EVERY CONFIGURED PLAYER
# ============================================================

if __name__ == "__main__":

    for dataset in DATASETS:

        player_name = (
            dataset["player_name"]
        )

        input_path = (
            RAW_DATA_DIR
            / dataset["input_filename"]
        )

        output_path = (
            PROCESSED_DIR
            / (
                f"competitive_matches_"
                f"{dataset['output_slug']}_cleaned.csv"
            )
        )

        clean_dataset(
            input_path=input_path,
            output_path=output_path,
            player_name=player_name,
        )

    print("\n" + "=" * 70)
    print("All configured datasets were cleaned successfully.")
    print("=" * 70)
