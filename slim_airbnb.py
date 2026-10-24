"""
Create a slim listings database for deployment.

    python slim_airbnb.py

The full airbnb.db is about 1 GB, because most of its 90 columns are long text
fields - description, amenities, host_about, image URLs. The agent never reads
those; it only needs the columns you can query on.

This writes airbnb_slim.db with just those columns, which lands in the tens of
megabytes - small enough for GitHub (100 MB limit) and fast to load.

The full airbnb.db is left untouched as your source of truth.
"""

import os
import sqlite3
import sys

SRC = "airbnb.db"
DST = "airbnb_slim.db"

# every column the agent might reasonably query on
KEEP = [
    "id", "name",
    "host_id", "host_name", "host_is_superhost", "host_listings_count",
    "host_identity_verified",
    "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms_text",
    "bedrooms", "beds",
    "price", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "license", "instant_bookable", "calculated_host_listings_count",
    "reviews_per_month",
    "estimated_occupancy_l365d", "estimated_revenue_l365d",
    "first_review", "last_review", "has_availability",
]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"Missing {SRC}. Run load_airbnb.py first.")

    src = sqlite3.connect(SRC)
    existing = [r[1] for r in src.execute("PRAGMA table_info(listings)")]
    keep = [c for c in KEEP if c in existing]
    skipped = [c for c in KEEP if c not in existing]

    print(f"Source : {SRC}  ({os.path.getsize(SRC) / 1e6:.0f} MB, {len(existing)} columns)")
    print(f"Keeping: {len(keep)} columns")
    if skipped:
        print(f"  (not in this scrape, skipped: {', '.join(skipped)})")

    if os.path.exists(DST):
        os.remove(DST)

    src.execute("ATTACH ? AS slim", (os.path.abspath(DST),))
    cols = ", ".join(f'"{c}"' for c in keep)
    src.execute(f"CREATE TABLE slim.listings AS SELECT {cols} FROM listings")
    src.commit()

    # keep the indexes the agent filters on
    for col in ["neighbourhood_cleansed", "neighbourhood_group_cleansed",
                "room_type", "price", "bedrooms"]:
        if col in keep:
            src.execute(f'CREATE INDEX slim.idx_{col} ON listings("{col}")')
    src.commit()
    src.close()

    size = os.path.getsize(DST) / 1e6
    print(f"\nWrote {DST}  ({size:.1f} MB)")
    print(f"  full db : {os.path.getsize(SRC) / 1e6:6.0f} MB")
    print(f"  slim db : {size:6.1f} MB")

    # sanity check
    con = sqlite3.connect(DST)
    n = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    med = con.execute("SELECT ROUND(AVG(price),2) FROM listings "
                      "WHERE price IS NOT NULL").fetchone()[0]
    con.close()
    print(f"\n  rows: {n:,}   avg price (where present): ${med}")
    print("\nDone. Deploy airbnb_slim.db instead of the full one.")


if __name__ == "__main__":
    main()
