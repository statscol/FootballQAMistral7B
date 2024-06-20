import pandas as pd
from sqlalchemy import create_engine

#Create DB for sqllite
ENGINE = create_engine("sqlite:///data/football.db")

DATA = {
    "goals": pd.read_csv("./data/Fifa_Results.csv"),
    "matches": pd.read_csv("./data/decision.csv"),
}


def modify_keys(df):
    """modifies home and away team for key creation"""
    return df.assign(
        home_team_key=df["home_team"].astype(str).str[:3],
        away_team_key=df["away_team"].astype(str).str[:3],
    )

def parse_date(df,col_name:str="date"):
    """parse str to datetime"""
    df.loc[:,col_name] = pd.to_datetime(df[col_name],format="%Y-%m-%d")
    return df

def create_key(df):
    """creates keys for football events using date, away and home team first 3 chars"""
    return df.assign(
        id=df["date"].astype(str).str.lower()
        + "-"
        + df["home_team_key"].str.lower()
        + "-"
        + df["away_team_key"].str.lower()
    )


if __name__ == "__main__":

    for table in DATA.keys():
        DATA[table]\
            .pipe(modify_keys)\
            .pipe(create_key)\
            .pipe(parse_date)\
            .drop(columns=["home_team_key", "away_team_key"])\
            .set_index("id")\
            .to_sql(table, ENGINE, if_exists="replace")
