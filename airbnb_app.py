"""
Airbnb Agent — Streamlit app.

One assistant, three tools:
  SQL   -> the listings database (price, rooms, neighbourhood, availability)
  DOCS  -> Airbnb's real policies (cancellation, refunds, house rules)
  BOTH  -> the database AND the policies together

Why BOTH is different here: in the job-market project, "both" meant SQL found
rows and RAG read their descriptions — the two tools shared one table. Here the
policy corpus is separate from the listings, so BOTH means: run the SQL, run
the policy search, and hand the model both answers to combine.

Run:   streamlit run airbnb_app.py
"""

import os
import re
import sqlite3
import time

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from openai import OpenAI

# prefer the slim database if it exists - it is what gets deployed
DB_PATH = "airbnb_slim.db" if os.path.exists("airbnb_slim.db") else "airbnb.db"
TABLE = "listings"
MODEL = "openai/gpt-oss-120b"
EMB_PATH = "policy_embeddings.npz"
TOP_K = 6          # policy chunks per question

st.set_page_config(page_title="Airbnb Agent", page_icon="🏠")


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def safe_md(text):
    return text.replace("$", r"\$")


def chat(prompt, tries=5):
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=get_api_key())
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


@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return TextEmbedding()


@st.cache_resource(show_spinner=False)
def get_policy_vectors():
    data = np.load(EMB_PATH, allow_pickle=True)
    vecs = data["vecs"].astype(float)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs, data["texts"], data["sources"], data["titles"]


# columns where the model must know the exact stored values
VALUE_COLS = [
    "room_type", "neighbourhood_group_cleansed", "property_type",
    "host_is_superhost", "instant_bookable", "has_availability",
]


@st.cache_data(show_spinner=False)
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
            note += (f"   -- {n} distinct neighbourhood names "
                     f"(e.g. 'Williamsburg', 'Harlem', 'Midtown')")
        lines.append(f"  {name} {ctype}{note}")
    con.close()
    return f"Table {TABLE} ({total:,} rows):\n" + "\n".join(lines)


@st.cache_data(show_spinner=False)
def db_stats():
    con = sqlite3.connect(DB_PATH)
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    con.close()
    return n


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
- availability_365 = nights available in the next 365 days.
- For "N-bedroom" questions you MUST add `bedrooms = N` AND
  `room_type = 'Entire home/apt'`. Private and shared rooms have bedrooms = 0,
  so without both filters the query returns studios and private rooms.
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
        events: "what is the cancellation policy", "can I get a refund",
        "are pets allowed", "how long do I have to report a problem".
BOTH -> needs the listings AND the policies together: "find a 2-bed in
        Brooklyn under $200 and tell me the cancellation policy",
        "which listings are cheap and refundable".
CHAT -> ONLY a greeting or small talk. If the message asks anything about
        listings or policies, use SQL, DOCS or BOTH.

Answer with exactly one word: SQL, DOCS, BOTH or CHAT.

Question: {question}""").upper()
    if "CHAT" in word:
        return "CHAT"
    if "BOTH" in word:
        return "BOTH"
    return "SQL" if "SQL" in word else "DOCS"


def sql_query(question, extra=""):
    """Write and run the query. Returns (markdown_table, sql, row_count)."""
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
    if len(rows) > 25:
        out += f"\n*... and {len(rows) - 25} more rows*"
    return out, sql, len(rows)


def policy_search(question):
    """Return the most relevant policy chunks as (context, source_list)."""
    model = get_model()
    vecs, texts, sources, titles = get_policy_vectors()

    q = np.asarray(list(model.query_embed([question]))[0], dtype=float)
    q = q / np.linalg.norm(q)
    scores = vecs @ q
    top = np.argsort(-scores)[:TOP_K]

    parts, src = [], []
    for i in top:
        parts.append(f"[{titles[i]}]\n{texts[i]}")
        src.append(f"**{titles[i]}**  —  {sources[i]}  (score {scores[i]:.3f})")
    return "\n\n".join(parts), "\n\n".join(src)


def run_sql(question):
    table, sql, _ = sql_query(question)
    return table, sql, ""


def run_docs(question):
    context, src = policy_search(question)
    answer = chat("""Answer using ONLY the Airbnb policy text below.
Quote the specific deadlines and numbers where the policy gives them.
Say you don't know ONLY if the text genuinely does not address the question -
if you found relevant text, answer with it and do NOT add a "don't know" note.

Policy text:
""" + context + f"\n\nQuestion: {question}")
    return answer, "", src


def run_both(question):
    table, sql, n = sql_query(question, extra=(
        "- This question ALSO asks about a policy. Write SQL for the LISTING "
        "part only (prices, rooms, locations). Do NOT try to select any "
        "policy information - that is looked up separately."))
    context, src = policy_search(question)
    answer = chat(f"""Answer the question using BOTH sources below.

1) A database result:
{table}

2) Relevant Airbnb policy text:
{context}

Use the database result for the listings and the policy text for the rules.
If one of them does not contain what is needed, say so. Mention any specific
deadlines or numbers from the policy.

IMPORTANT: the database result is usually limited to the TOP few rows. If it
shows a single listing, that means "the best match", NOT "the only one that
exists". Never claim a listing is the only one unless the result actually
counted them.

Question: {question}""")
    return answer, sql, src


# ---------------- UI ----------------

st.title("Airbnb Agent")
st.caption("One assistant, three tools. It decides whether your question needs the "
           "listings database (SQL), Airbnb's policies (RAG), or both.")

if not os.path.exists(DB_PATH) or not os.path.exists(EMB_PATH):
    st.error("Missing `airbnb.db` or `policy_embeddings.npz` in the app folder.")
    st.stop()

with st.sidebar:
    st.subheader("What's inside")
    st.write(f"**{db_stats():,}** NYC listings (SQL tool)")
    st.write("**5** Airbnb policies, 47 chunks (RAG tool)")
    st.divider()
    st.caption("Try:")
    st.caption("• How many listings are in Brooklyn?  (SQL)")
    st.caption("• What is the median price of an entire home?  (SQL)")
    st.caption("• How long do I have to report a problem?  (RAG)")
    st.caption("• Are pets allowed?  (RAG)")
    st.caption("• Cheapest 2-bedroom in Manhattan + the cancellation policy  (BOTH)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(safe_md(m["content"]))
        if m.get("detail"):
            with st.expander(m.get("detail_label", "Details")):
                st.markdown(safe_md(m["detail"]))

if question := st.chat_input("Ask about NYC listings or Airbnb's policies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(safe_md(question))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tool = choose_tool(question)
            if tool == "CHAT":
                answer = chat(
                    f"Today is {__import__('datetime').date.today().strftime('%A, %d %B %Y')}.\n"
                    f"The user said: {question}\n"
                    "You are the Airbnb Assistant (that is your name), a demo "
                    "built as a learning project and NOT affiliated with Airbnb. "
                    "Reply in ONE "
                    "friendly sentence. Do NOT state any facts about listings or "
                    "policies - you have no data for this message. Then invite them "
                    "to ask about NYC listings or Airbnb's policies.")
                detail, label = "", "Details"
            elif tool == "SQL":
                answer, sql, _ = run_sql(question)
                detail, label = f"```sql\n{sql}\n```", "SQL it ran"
            elif tool == "BOTH":
                answer, sql, src = run_both(question)
                detail = f"```sql\n{sql}\n```\n\n**Policy chunks it read:**\n\n{src}"
                label = "SQL + policy used"
            else:
                answer, _, src = run_docs(question)
                detail, label = src, "Policy chunks it used"

        st.markdown(f"`tool: {tool}`")
        st.markdown(safe_md(answer))
        if detail:
            with st.expander(label):
                st.markdown(safe_md(detail))

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "detail": detail, "detail_label": label,
    })
