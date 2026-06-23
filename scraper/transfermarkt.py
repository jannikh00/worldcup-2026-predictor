"""
scrape_transfermarkt.py
=======================
Pipeline Step 2 — collect squad data (age + market value) per national team,
for each season 2022..2026, via an AI model on the NRP (OpenAI-compatible) endpoint.

DIVISION OF LABOUR
------------------
  * The AI model does the EXTRACTION: HTML -> a clean JSON list of players.
    (No BeautifulSoup selectors that break when the page layout changes.)
  * Plain Python does the MATH: mean / std of age and value per (country, year).
    LLMs are unreliable at arithmetic, so aggregation stays in code.

NRP ENDPOINT (OpenAI-compatible)
--------------------------------
  base_url : https://ellm.nrp-nautilus.io/v1
  api_key  : a token you mint at the NRP LLM token page (store in env NRP_API_KEY)
  model    : "gpt-oss" (stable, good for extraction) or "qwen3" (long context)
  Note: some models "think" by default; we disable that for clean JSON + speed.

GETTING THE HTML
----------------
Transfermarkt blocks bots, so requests may 403. The script DECOUPLES fetching
from extraction:
  - It first looks for a saved page at  data/pages/<country>_<year>.html
    (open the squad page in your browser, Save As -> that folder).
  - Otherwise it falls back to requests.
The AI-extraction step is identical either way.

INPUT
-----
data/teams.csv  with columns:  country,slug,team_id
  country -> the name you'll join on later (stay consistent)
  slug    -> the URL slug, e.g. "brasilien"
  team_id -> the Transfermarkt club/verein id, e.g. 3439
  The script builds:  https://www.transfermarkt.com/<slug>/kader/verein/<team_id>/saison_id/<year>
  (Open a national team -> Squad -> read slug and id straight from the URL.)

OUTPUT
------
data/players_raw.csv    one row per (player, year)
data/team_features.csv  one row per (country, year): age_mean/std, value_mean/std

Run:  python3 scrape_transfermarkt.py
"""

import os
import re
import csv
import json
import time

import pandas as pd
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------------- Config -----------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
TEAMS_CSV     = os.path.join(_HERE, "../data/teams.csv")    # input: country,slug,team_id
PAGES_DIR     = os.path.join(_HERE, "../data/pages")        # optional saved HTML: <country>_<year>.html
PLAYERS_CSV   = os.path.join(_HERE, "../data/players_raw.csv")   # output: one row per (player, year)
FEATURES_CSV  = os.path.join(_HERE, "../data/team_features.csv") # output: one row per (country, year)

YEARS         = [2022, 2023, 2024, 2025, 2026]   # saison_id values to collect
BASE_URL      = "https://ellm.nrp-nautilus.io/v1"
MODEL         = "gpt-oss"                      # or "qwen3" for longer context
DISABLE_THINKING = True                        # strip model "thinking" for clean JSON
REQUEST_PAUSE = 1.5                            # seconds between network fetches (be polite)

# Instruction the AI follows to turn raw HTML into structured player rows.
EXTRACTION_SYSTEM_PROMPT = """You are a precise data-extraction tool.
You will be given the HTML of a Transfermarkt national-team squad page.
Extract EVERY player listed in the squad table. For each player output an object:
  - "name": player name (string)
  - "age": age in years (integer)
  - "market_value_eur": market value in whole euros as an integer
        examples: "€80.00m" -> 80000000 , "€1.50m" -> 1500000 ,
                  "€900k" -> 900000 , "€500Th." -> 500000 ;
        if no value is shown ("-" or empty) use null
Return ONLY a JSON array of these objects. No prose, no markdown, no code fences."""

# OpenAI-compatible client pointed at NRP. Token comes from env NRP_API_KEY.
client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("NRP_API_KEY", ""))


# --------------------------- Helper: value parsing ------------------------
def coerce_value(v):
    """Make a market value into an integer number of euros. The AI is asked to
    return a number already; this is a safety net for raw strings like '€80.00m'.
    Returns None if unknown."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower().replace("€", "").replace(",", "").replace(" ", "")
    if s in ("", "-", "none", "null"):
        return None
    m = re.match(r"^([0-9.]+)\s*(bn|m|k|th\.?)?$", s)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2) or ""
    factor = {"bn": 1_000_000_000, "m": 1_000_000,
              "k": 1_000, "th": 1_000, "th.": 1_000, "": 1}[unit]
    return int(num * factor)


# --------------------------- Helper: parse AI output ----------------------
def parse_player_json(text):
    """Turn the model's text response into a list of player dicts. Strips code
    fences, and if the model emitted reasoning before the JSON, grabs the
    outermost [...] array."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", cleaned, flags=re.S)   # find the JSON array anywhere
        if not m:
            raise
        data = json.loads(m.group(0))
    players = []
    for p in data:
        players.append({
            "name": p.get("name"),
            "age": p.get("age"),
            "market_value_eur": coerce_value(p.get("market_value_eur")),
        })
    return players


# --------------------------- Helper: get the HTML -------------------------
def clean_html(html, max_chars=80000):
    """Strip <script>/<style> and collapse whitespace to cut token cost, then cap
    length. Light cleaning, not data extraction."""
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"\s+", " ", html)
    return html[:max_chars]


def build_url(slug, team_id, year):
    return f"https://www.transfermarkt.us/{slug}/kader/verein/{team_id}/saison_id/{year}"


def get_html(country, year, url):
    """Prefer a locally saved page (beats anti-bot); otherwise fetch with requests."""
    local = os.path.join(PAGES_DIR, f"{country}_{year}.html")
    if os.path.exists(local):
        print(f"    using saved page {local}")
        with open(local, encoding="utf-8", errors="ignore") as f:
            return f.read()
    print(f"    fetching {url}")
    headers = {  # look like a real browser
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(REQUEST_PAUSE)
    return r.text


# --------------------------- Helper: AI extraction ------------------------
def extract_players(html):
    """Send cleaned HTML to the NRP model and get back a list of player dicts."""
    kwargs = dict(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": clean_html(html)},
        ],
    )
    # Disable "thinking" on models that support the flag (qwen3 etc.) for clean JSON.
    if DISABLE_THINKING:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("extra_body", None)      # some models reject the flag; retry once
        resp = client.chat.completions.create(**kwargs)
    return parse_player_json(resp.choices[0].message.content)


# ------------------------------- Aggregation ------------------------------
def aggregate(players_df):
    """One row per (country, year): mean/std of age and of market value.
    Population std (ddof=0) because a squad is the WHOLE group, not a sample.
    Players with no listed value are dropped from the value stats."""
    rows = []
    for (country, year), g in players_df.groupby(["country", "year"]):
        ages = g["age"].dropna()
        vals = g["market_value_eur"].dropna()
        rows.append({
            "country": country,
            "year": year,
            "n_players": len(g),
            "age_mean": round(ages.mean(), 2) if len(ages) else None,
            "age_std": round(ages.std(ddof=0), 2) if len(ages) else None,
            "value_mean": round(vals.mean(), 2) if len(vals) else None,
            "value_std": round(vals.std(ddof=0), 2) if len(vals) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------- Main ----------------------------------
def main():
    teams = list(csv.DictReader(open(TEAMS_CSV, encoding="utf-8")))
    print(f"Loaded {len(teams)} teams from {TEAMS_CSV}; collecting years {YEARS}")

    all_players = []
    for i, row in enumerate(teams, 1):
        country = row["country"].strip()
        slug, team_id = row["slug"].strip(), row["team_id"].strip()
        print(f"[{i}/{len(teams)}] {country}")
        for year in YEARS:
            url = build_url(slug, team_id, year)
            try:
                html = get_html(country, year, url)
                players = extract_players(html)
                for p in players:
                    p["country"], p["year"] = country, year
                all_players.extend(players)
                print(f"    {year}: extracted {len(players)} players")
            except Exception as e:
                # keep going; a missing season (e.g. 2026 not published yet) is fine
                print(f"    {year}: FAILED ({e})")

    if not all_players:
        raise SystemExit("No players extracted - check teams.csv / saved pages / NRP token.")

    # save raw per-player data
    players_df = pd.DataFrame(all_players)
    os.makedirs(os.path.dirname(PLAYERS_CSV), exist_ok=True)
    players_df.to_csv(PLAYERS_CSV, index=False)
    print(f"\nSaved {len(players_df)} player rows -> {PLAYERS_CSV}")

    # aggregate to per-(country, year) features
    features_df = aggregate(players_df)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(features_df)} (country, year) rows -> {FEATURES_CSV}\n")
    print("Team features preview:")
    print(features_df.to_string(index=False))


if __name__ == "__main__":
    main()