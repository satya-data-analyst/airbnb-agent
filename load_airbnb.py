"""
Load the Inside Airbnb listings file into a SQLite database.

    python load_airbnb.py

Looks for the listings file in a few likely places, cleans it, loads it into
airbnb.db, and prints a summary so we can see what we actually have.

The loader is schema-agnostic: it loads whatever columns the file has, so it
works for any city or scrape date.
"""

import os
import re
import sqlite3
import sys

import pandas as pd

CANDIDATES = [
    "data/listings.csv.gz",
    "listings.csv.gz",
    "data/listings.csv",
    "listings.csv",
]
DB_PATH = "airbnb.db"
TABLE = "listings"


def find_file():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit(
        "Could not find the listings file. Put listings.csv.gz in this folder "
        "(or in a data/ subfolder) and run again."
    )


def clean_price(value):
    """'$1,200.00' -> 1200.0 ; '' or NaN -> None"""
    if pd.isna(value):
        return None
    digits = re.sub(r"[^0-9.]", "", str(value))
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def main():
    path = find_file()
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  rows: {len(df):,}   columns: {len(df.columns)}")

    # price arrives as text like "$1,200.00"
    if "price" in df.columns:
        df["price"] = df["price"].apply(clean_price)
        print(f"  price: parsed -> {df['price'].notna().sum():,} values, "
              f"median ${df['price'].median():,.0f}")

    # keep dates as plain strings so SQLite is easy to query
    for col in ["host_since", "first_review", "last_review", "last_scraped",
                "calendar_last_scraped"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype(str)

    con = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE, con, if_exists="replace", index=False)

    # an index on the columns we'll filter on most
    for col in ["neighbourhood_cleansed", "room_type", "price"]:
        if col in df.columns:
            con.execute(f'CREATE INDEX IF NOT EXISTS idx_{col} '
                        f'ON {TABLE}("{col}")')
    con.commit()
    con.close()
    print(f"\nWrote {DB_PATH}  (table: {TABLE})")

    # ---- show what we have ----
    cols = list(df.columns)
    print(f"\nColumns ({len(cols)}):")
    for i in range(0, len(cols), 4):
        print("  " + " | ".join(f"{c[:24]:<24}" for c in cols[i:i + 4]))

    if "room_type" in df.columns:
        print("\nListings by room type:")
        print(df["room_type"].value_counts().to_string())

    ncol = next((c for c in ["neighbourhood_cleansed", "neighbourhood",
                             "neighbourhood_group_cleansed"] if c in df.columns), None)
    if ncol:
        print(f"\nTop 10 neighbourhoods by listing count ({ncol}):")
        print(df[ncol].value_counts().head(10).to_string())

    if "price" in df.columns and ncol:
        print(f"\nMedian price by neighbourhood ({ncol}, top 8 by count):")
        top = df[ncol].value_counts().head(8).index
        med = df[df[ncol].isin(top)].groupby(ncol)["price"].median().round(0)
        print(med.sort_values(ascending=False).to_string())

    print("\nDone. Next: we point the agent at airbnb.db.")


if __name__ == "__main__":
    main()
