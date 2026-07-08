"""
generate_dataset.py — synthetic dataset generator
Run this to create or expand emails.json using Claude.

Usage:
  python data/generate_dataset.py --n 10 --category billing
  python data/generate_dataset.py --n 25   # generates 25 new examples across all categories
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CATEGORIES = ["billing", "technical", "account", "feature_request", "complaint", "inquiry"]

CATEGORY_HINTS = {
    "billing":         "invoice, charge, refund, plan upgrade, pricing, payment failure",
    "technical":       "bug, sync issue, login, crash, integration broken, notification",
    "account":         "add user, setup inbox, transfer ownership, permissions, onboarding",
    "feature_request": "new feature, integration, API, export, customisation request",
    "complaint":       "frustrated customer, repeated issue, slow support, outage impact",
    "inquiry":         "pre-sales question, compatibility, limits, security, GDPR",
}


def generate_pairs(client: Anthropic, category: str, n: int, start_id: int) -> list[dict]:
    hint = CATEGORY_HINTS.get(category, "")
    prompt = f"""Generate {n} realistic B2B customer support email pairs for a company called Hiver (a shared inbox tool for Gmail).

Category: {category}
Topics to cover: {hint}

Each pair must be distinct. Use realistic names, companies, and specific details — avoid generic placeholder text.

Reply ONLY with a JSON array (no markdown). Each item:
{{
  "id": <integer starting at {start_id}>,
  "category": "{category}",
  "subject": "<email subject>",
  "incoming_email": "<customer email body, 3-6 sentences>",
  "reply": "<support reply, 4-8 sentences, professional and solution-focused>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Total pairs to generate")
    parser.add_argument("--category", default=None, help="Single category; omit for all")
    parser.add_argument("--output", default="data/emails.json")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    client = Anthropic()
    out_path = Path(args.output)

    existing = []
    if args.append and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    start_id = max((e["id"] for e in existing), default=0) + 1
    categories = [args.category] if args.category else CATEGORIES

    new_pairs: list[dict] = []
    per_cat = max(1, args.n // len(categories))

    for cat in categories:
        n_cat = per_cat if cat != categories[-1] else args.n - len(new_pairs)
        if n_cat <= 0:
            break
        print(f"Generating {n_cat} × {cat}…")
        pairs = generate_pairs(client, cat, n_cat, start_id + len(new_pairs))
        new_pairs.extend(pairs)

    all_pairs = existing + new_pairs
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"Done — {len(all_pairs)} total pairs written to {out_path}")


if __name__ == "__main__":
    main()
