"""
Evaluate the Airbnb agent against a golden set.

    python evaluate_airbnb.py

Scores two things:
  1. Routing accuracy - did it pick the right tool (SQL / DOCS / BOTH / CHAT)?
  2. Answer accuracy  - does the answer contain the expected fact?

Expected values in airbnb_golden_set.json come from the real data, so this
measures the agent, not my guesses.

Note: the routing and tool prompts are duplicated from airbnb_app.py on purpose,
so this script runs standalone without Streamlit. If you change one, change both.
"""

import json
import os
import re
import sqlite3
import time

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

# use the slim database if present, so this matches the deployed app
DB_PATH = "airbnb_slim.db" if os.path.exists("airbnb_slim.db") else "airbnb.db"
TABLE = "listings"
MODEL = "openai/gpt-oss-120b"
EMB_PATH = "policy_embeddings.npz"
TOP_K = 6
GOLDEN = "airbnb_golden_set.json"


def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No API key. Set GROQ_API_KEY or create a .env file.")
    return key


client = None


def chat(prompt, tries=5):
    for attempt in range(tries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "rate limit" in msg) and attempt < tries - 1:
                time.sleep(6 * (attempt + 1))
                continue
            raise


def clean_sql(text):
    return re.sub(r"^```(?:sql)?|```$", "", text, flags=re.MULTILINE).strip()


VALUE_COLS = ["room_type", "neighbourhood_group_cleansed", "property_type",
              "host_is_superhost", "instant_bookable", "has_availability"]


def get_schema():
    con = sqlite3.connect(DB_PATH)
    total = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    lines = []
    for _, name, ctype, *_ in con.execute(f"PRAGMA table_info({TABLE})"):
        nulls = con.execute(
            f'SELECT COUNT(*) FROM {TABLE} WHERE "{name}" IS NULL').fetchone()[0]
        pct = 100 * nulls / total if total else 0
        note = f"   -- {pct:.0f}% empty" if pct >= 5 else ""
        if name in VALUE_COLS:
            vals = [str(r[0]) for r in con.execute(
                f'SELECT DISTINCT "{name}" FROM {TABLE} '
                f'WHERE "{name}" IS NOT NULL LIMIT 12')]
            if vals:
                note += "   -- values: " + ", ".join(f"'{v}'" for v in vals)
        elif name == "neighbourhood_cleansed":
            n = con.execute(
                f'SELECT COUNT(DISTINCT "{name}") FROM {TABLE}').fetchone()[0]
            note += f"   -- {n} distinct neighbourhood names"
        lines.append(f"  {name} {ctype}{note}")
    con.close()
    return f"Table {TABLE} ({total:,} rows):\n" + "\n".join(lines)


SQL_RULES = """Rules:
- Only a single SELECT. Never modify data.
- The database holds LISTING data only. There is NO column about policies,
  cancellations, refunds or rules. Never select such a column.
- price: about a third of listings have NO price. ALWAYS add `price IS NOT NULL`
  to any query that uses price, or the numbers will be wrong.
- BOROUGHS (Brooklyn, Manhattan, Queens, Bronx, Staten Island) are in
  `neighbourhood_group_cleansed`. The plain `neighbourhood` column is mostly
  empty - NEVER use it.
- For smaller areas (Williamsburg, Harlem, Midtown...) use `neighbourhood_cleansed`.
- Use the EXACT stored value shown in the schema, e.g. room_type = 'Entire home/apt'.
  Values are CASE-SENSITIVE, so copy them exactly as listed.
- For "N-bedroom" questions you MUST add `bedrooms = N` AND
  `room_type = 'Entire home/apt'`.
- SQLite has NO median function. For a median, use a window function like this:
  SELECT AVG(price) AS median_price FROM (
    SELECT price, ROW_NUMBER() OVER (ORDER BY price) AS rn,
           COUNT(*) OVER () AS n
    FROM listings WHERE <filters> AND price IS NOT NULL
  ) WHERE rn IN ((n + 1) / 2, (n + 2) / 2);
  Never use AVG() as a substitute for a median.
- Return ONLY the SQL, no explanation, no markdown."""


def choose_tool(question):
    word = chat(f"""You route questions to one of four tools.

SQL  -> numbers, counts, prices, neighbourhoods, room types, availability,
        reviews, hosts: "how many", "average price", "cheapest", "which
        neighbourhood", "listings with 3 bedrooms".
DOCS -> what Airbnb's policies SAY: cancellations, refunds, deadlines,
        house rules, pets, smoking, damage, host cancellations, disruptive
        events.
BOTH -> needs the listings AND the policies together.
CHAT -> ONLY a greeting or small talk.

Answer with exactly one word: SQL, DOCS, BOTH or CHAT.

Question: {question}""").upper()
    if "CHAT" in word:
        return "CHAT"
    if "BOTH" in word:
        return "BOTH"
    return "SQL" if "SQL" in word else "DOCS"


def sql_query(question, extra=""):
    sql = clean_sql(chat(f"""You are a SQLite expert. Write ONE SQLite query to answer the question.

{get_schema()}

{SQL_RULES}
{extra}

Question: {question}"""))
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql).fetchall()
    except Exception as e:
        con.close()
        return f"Query failed: {e}", sql, 0
    con.close()
    if not rows:
        return "The query returned no rows.", sql, 0
    cols = list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n"
    out += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for r in rows[:25]:
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out, sql, len(rows)


def policy_search(question, model, vecs, texts, sources, titles):
    q = np.asarray(list(model.query_embed([question]))[0], dtype=float)
    q = q / np.linalg.norm(q)
    scores = vecs @ q
    top = np.argsort(-scores)[:TOP_K]
    parts, src = [], []
    for i in top:
        parts.append(f"[{titles[i]}]\n{texts[i]}")
        src.append(f"**{titles[i]}** — {sources[i]}")
    return "\n\n".join(parts), "\n\n".join(src)


def run_one(item, model, vecs, texts, sources, titles):
    q = item["question"]
    tool = choose_tool(q)

    if tool == "CHAT":
        return tool, "(chat reply)"

    if tool == "SQL":
        table, _, _ = sql_query(q)
        return tool, table

    context, _ = policy_search(q, model, vecs, texts, sources, titles)

    if tool == "DOCS":
        answer = chat("""Answer using ONLY the Airbnb policy text below.
Quote the specific deadlines and numbers where the policy gives them.
Say you don't know ONLY if the text genuinely does not address the question -
if you found relevant text, answer with it and do NOT add a "don't know" note.

Policy text:
""" + context + f"\n\nQuestion: {q}")
        return tool, answer

    # BOTH
    table, _, _ = sql_query(q, extra=(
        "- This question ALSO asks about a policy. Write SQL for the LISTING "
        "part only. Do NOT try to select any policy information."))
    answer = chat(f"""Answer the question using BOTH sources below.

1) A database result:
{table}

2) Relevant Airbnb policy text:
{context}

Use the database result for the listings and the policy text for the rules.
IMPORTANT: the database result is usually limited to the TOP few rows. If it
shows a single listing, that means "the best match", NOT "the only one that
exists".

Question: {q}""")
    return tool, answer


def score_answer(answer, expect_any):
    if not expect_any:
        return None
    low = answer.lower()
    return any(k.lower() in low for k in expect_any)


def main():
    global client
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=get_api_key())

    data = np.load(EMB_PATH, allow_pickle=True)
    vecs = data["vecs"].astype(float)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    texts, sources, titles = data["texts"], data["sources"], data["titles"]
    model = TextEmbedding()

    results = []
    for i, item in enumerate(golden, 1):
        if i > 1:
            time.sleep(6)      # stay under the free-tier tokens-per-minute cap
        tool, answer = run_one(item, model, vecs, texts, sources, titles)
        tool_ok = tool == item["expect_tool"]
        ans_ok = score_answer(answer, item.get("expect_any"))
        results.append({**item, "got_tool": tool, "tool_ok": tool_ok,
                        "answer_ok": ans_ok, "answer": answer[:300]})

        mark = "OK  " if tool_ok else "MISS"
        ans_mark = "" if ans_ok is None else ("  ans:OK" if ans_ok else "  ans:MISS")
        print(f"{i:2d}. [{mark}]{ans_mark}  {item['question'][:56]}")
        if not tool_ok:
            print(f"       expected {item['expect_tool']}, got {tool}")

    n = len(results)
    tool_acc = sum(r["tool_ok"] for r in results) / n
    scored = [r for r in results if r["answer_ok"] is not None]
    ans_acc = (sum(r["answer_ok"] for r in scored) / len(scored)) if scored else 0.0

    print("\n" + "=" * 52)
    print(f"  Routing accuracy : {tool_acc:.0%}   "
          f"({sum(r['tool_ok'] for r in results)}/{n})")
    print(f"  Answer accuracy  : {ans_acc:.0%}   "
          f"({sum(r['answer_ok'] for r in scored)}/{len(scored)})")
    print("=" * 52)

    json.dump(results, open("airbnb_eval_results.json", "w", encoding="utf-8"), indent=2)
    print("Saved airbnb_eval_results.json")


if __name__ == "__main__":
    main()
