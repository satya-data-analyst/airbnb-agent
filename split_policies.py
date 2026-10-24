"""
Split the pasted Airbnb policy text into one clean file per policy.

    python split_policies.py

Reads airbnb.txt (the blob you dragged from the website) and writes:

    data/policies/01_rebooking_and_refund.txt
    data/policies/02_cancellation_policies.txt
    data/policies/03_ground_rules_for_guests.txt
    data/policies/04_major_disruptive_events.txt
    data/policies/05_host_cancellation.txt

Why split? RAG works far better with one topic per document. A single blob
means every question retrieves the whole 31 KB, and the model has to find the
needle itself. Separate files mean a question about cancellation windows
retrieves the cancellation policy — and nothing else.

The headings in the pasted text sit either at the start of a line or at the
END of a line (the drag-copy merged them onto the previous paragraph), so the
splitter looks for both patterns.
"""

import os
import re
import sys

SOURCE = "airbnb.txt"
OUT_DIR = "data/policies"

# (heading as it appears in the text, output filename, human title)
POLICIES = [
    ("Rebooking and refund policy for homes", "01_rebooking_and_refund.txt",
     "Rebooking and Refund Policy for Homes"),
    ("Cancellation policies for your home", "02_cancellation_policies.txt",
     "Cancellation Policies for Your Home"),
    ("Ground rules for home guests", "03_ground_rules_for_guests.txt",
     "Ground Rules for Home Guests"),
    ("Major Disruptive Events Policy", "04_major_disruptive_events.txt",
     "Major Disruptive Events Policy"),
    ("Host Cancellation Policy for homes", "05_host_cancellation.txt",
     "Host Cancellation Policy for Homes"),
]


def find_source():
    for candidate in (SOURCE, "data/" + SOURCE, "airbnb.txt.txt"):
        if os.path.exists(candidate):
            return candidate
    sys.exit(
        f"Could not find {SOURCE}. Put it in this folder (the file you pasted "
        f"the Airbnb text into) and run again."
    )


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)     # trailing spaces
    text = re.sub(r"[ \t]{3,}", " ", text)     # the big gaps left by the drag
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank runs
    return text.strip()


def strip_trailing_chatter(text):
    """Drop a short note that got pasted onto the end of the last policy."""
    q = text.find("?")
    if q == -1:
        return text
    cut = text.rfind(".", 0, q)
    if cut == -1:
        return text
    tail = text[cut + 1:]
    if len(tail.split()) < 40:          # a short trailing note, not policy text
        return text[:cut + 1].rstrip()
    return text


def locate(text, heading):
    """Find the heading where it starts a line, or ends one."""
    for m in re.finditer(re.escape(heading), text):
        start, end = m.start(), m.end()
        at_line_start = start == 0 or text[start - 1] == "\n"
        at_line_end = end >= len(text) or text[end] == "\n"
        if at_line_start or at_line_end:
            return start
    return None


def main():
    path = find_source()
    raw = open(path, encoding="utf-8").read()
    print(f"Read {path}  ({len(raw):,} characters)")

    marks = []
    for heading, filename, title in POLICIES:
        pos = locate(raw, heading)
        if pos is None:
            print(f"  ! heading not found: {heading!r}")
        else:
            marks.append((pos, heading, filename, title))

    marks.sort()
    if len(marks) < 2:
        sys.exit("Could not find enough headings — check the source file.")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nFound {len(marks)} policies:\n")
    for i, (pos, heading, filename, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(raw)
        body = strip_trailing_chatter(clean(raw[pos:end]))
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        words = len(body.split())
        print(f"  {filename:<34} {words:>6,} words   \"{title}\"")

    print(f"\nWrote {len(marks)} files to {OUT_DIR}/")
    print("Next: we chunk and embed these for the RAG tool.")


if __name__ == "__main__":
    main()
