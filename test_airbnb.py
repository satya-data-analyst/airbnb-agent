"""
Automated tests for the Airbnb Agent.

Run with:   pytest -q

These need NO API key and NO internet — they check the data, the policy corpus,
the golden set and the scoring logic, so CI can run them on every push.
"""

import glob
import json
import os
import sqlite3

import numpy as np
import pytest

DB = "airbnb_slim.db" if os.path.exists("airbnb_slim.db") else "airbnb.db"

EXPECTED_COLUMNS = [
    "id", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "room_type", "price", "bedrooms", "accommodates",
    "availability_365", "number_of_reviews", "review_scores_rating",
]

ROUTES = {"SQL", "DOCS", "BOTH", "CHAT"}


def test_database_exists():
    assert os.path.exists(DB), f"{DB} is missing"


def test_database_has_expected_columns():
    con = sqlite3.connect(DB)
    cols = {row[1] for row in con.execute("PRAGMA table_info(listings)")}
    con.close()
    for c in EXPECTED_COLUMNS:
        assert c in cols, f"missing column: {c}"


def test_database_row_count():
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    con.close()
    assert n == 29918, f"expected 29,918 listings, found {n}"


def test_borough_values_are_the_ones_we_route_on():
    con = sqlite3.connect(DB)
    vals = {r[0] for r in con.execute(
        "SELECT DISTINCT neighbourhood_group_cleansed FROM listings")}
    con.close()
    assert "Brooklyn" in vals and "Manhattan" in vals, f"unexpected boroughs: {vals}"


def test_price_is_often_missing_so_queries_must_filter():
    """The agent's SQL rules rely on this: price is missing for many listings."""
    con = sqlite3.connect(DB)
    total = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    with_price = con.execute(
        "SELECT COUNT(*) FROM listings WHERE price IS NOT NULL").fetchone()[0]
    con.close()
    assert 0 < with_price < total, "price should be present for some but not all rows"


def test_policy_files_present():
    files = sorted(glob.glob("data/policies/*.txt"))
    assert len(files) == 5, f"expected 5 policy files, found {len(files)}"
    for f in files:
        assert os.path.getsize(f) > 500, f"{f} looks empty"


def test_policy_embeddings_load():
    data = np.load("policy_embeddings.npz", allow_pickle=True)
    vecs = data["vecs"]
    assert vecs.shape[1] == 384, f"unexpected embedding size {vecs.shape[1]}"
    assert len(data["texts"]) == len(vecs) == len(data["sources"])
    assert len(vecs) >= 20, "too few policy chunks"


def test_golden_set_is_wellformed():
    golden = json.load(open("airbnb_golden_set.json", encoding="utf-8"))
    assert len(golden) >= 12, "golden set is too small"
    for item in golden:
        assert item["question"].strip()
        assert item["expect_tool"] in ROUTES
        assert "expect_any" in item
        assert "category" in item


def test_golden_set_covers_all_routes():
    golden = json.load(open("airbnb_golden_set.json", encoding="utf-8"))
    tools = {i["expect_tool"] for i in golden}
    assert ROUTES <= tools, f"golden set misses routes: {ROUTES - tools}"


def test_golden_set_has_a_negative_test():
    golden = json.load(open("airbnb_golden_set.json", encoding="utf-8"))
    assert any(i["category"] == "rag-negative" for i in golden), \
        "keep at least one question the policies cannot answer"


def test_scoring_logic():
    ev = pytest.importorskip("evaluate_airbnb")
    assert ev.score_answer("Brooklyn has 10,290 listings", ["10290", "10,290"]) is True
    assert ev.score_answer("no idea", ["10290"]) is False
    assert ev.score_answer("anything at all", []) is None
