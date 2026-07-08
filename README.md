# Hiver — AI Email Response System

An end-to-end system that generates and evaluates AI-suggested replies for customer support emails, built with Python + Claude.

---

## What's built

| Component | File | Description |
|-----------|------|-------------|
| Dataset | `data/emails.json` | 25 realistic B2B support email pairs across 6 categories |
| Dataset generator | `data/generate_dataset.py` | Script to create more pairs via Claude |
| Response generator | `generator.py` | TF-IDF RAG + Claude Sonnet for reply generation |
| Evaluator | `evaluator.py` | ROUGE + Semantic Similarity + LLM Judge → composite score |
| Demo runner | `main.py` | CLI to run and display results end-to-end |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the demo (5 held-out test emails)

```bash
python main.py
```

### Other modes

```bash
# Test a specific email by ID
python main.py --id 3

# Evaluate all 25 emails in the dataset
python main.py --all

# Type a custom email and get a generated reply + evaluation
python main.py --custom

# Generate 10 more synthetic examples in the billing category
python data/generate_dataset.py --n 10 --category billing --append
```

---

## 1. Dataset

**Source:** Synthetic, hand-reviewed email pairs modelled on real B2B SaaS customer support patterns, specifically tailored to Hiver's product domain (shared Gmail inboxes, assignment rules, canned responses, billing, onboarding).

**Size:** 25 pairs across 6 categories:
- `billing` — invoices, duplicate charges, upgrades, refunds, enterprise pricing
- `technical` — Gmail sync, login failures, mobile crashes, notification bugs
- `account` — user onboarding, inbox setup, ownership transfer, seat management
- `feature_request` — Slack integration, analytics, API, bulk import
- `complaint` — repeated issues, outage impact, slow support SLA
- `inquiry` — Outlook compatibility, user limits, GDPR compliance

**Why representative:** The categories mirror the real distribution of support queries for B2B SaaS tools. The emails use realistic names, specific error codes/ticket IDs, and company names — they are not generic placeholder text. Each reply models the tone and structure expected from a professional support team.

**How to expand:** Run `python data/generate_dataset.py --n 50 --append` to generate 50 more synthetic pairs using Claude.

---

## 2. Response Generator

**Architecture: TF-IDF Retrieval + Few-Shot Prompting**

```
Incoming email
     │
     ▼
TF-IDF vectorizer (bigrams, stop-words removed)
     │
     ▼
cosine_similarity → top-3 most similar past emails (subject + body)
     │
     ▼
Few-shot prompt → Claude Sonnet 4.6 → generated reply
```

**Why this approach:**

| Approach | Trade-offs | Why I chose this |
|----------|-----------|-----------------|
| Zero-shot prompting | Fast, no retrieval needed | Doesn't learn from past replies; inconsistent tone |
| TF-IDF RAG (chosen) | Lightweight, no embedding API cost, interpretable | Grounded in real examples; works offline; fast |
| Dense embeddings RAG | More semantically aware | Adds latency + cost; overkill for 25 examples |
| Fine-tuning | Best quality if data is large enough | Requires 100s–1000s of examples; impractical here |

**Key design decisions:**
- Bigram TF-IDF (`ngram_range=(1,2)`) captures phrases like "shared inbox" and "billing issue" better than unigrams alone
- Top-3 examples provide enough context without overloading the prompt
- The prompt explicitly instructs Claude to match tone from the examples — this is the key mechanism that grounds the generation in past behaviour

---

## 3. Accuracy System

### Why "accuracy" is hard for email replies

Exact match is wrong. Two replies can be equally correct while sharing zero words. BLEU (designed for translation) punishes length variation. Standard classification metrics don't apply to free-form text.

**Good "accuracy" for a customer support reply means:**
1. It answers what was actually asked (relevance)
2. It uses the right tone — empathetic but professional (tone)
3. It either solves the problem or gives clear next steps (completeness)
4. The information is correct (accuracy)
5. It's easy to understand (clarity)

### Metrics used

#### ROUGE-1 / ROUGE-2 / ROUGE-L
Measures n-gram overlap between the generated reply and the reference (human) reply.

**Strength:** Cheap, fast, interpretable.  
**Weakness:** Blind to paraphrasing. A perfectly valid reply that uses different words will score near zero.  
**Role here:** Sanity check — if ROUGE is very low, the reply may be off-topic.

#### Semantic Similarity (Sentence-BERT cosine)
Encodes both replies with `all-MiniLM-L6-v2` and measures cosine distance in embedding space.

**Strength:** Captures *meaning*, not just word choice. "Your refund is processed" and "We've sent back your money" score high.  
**Weakness:** Doesn't tell us if the reply is professionally written or empathetic.  
**Role here:** Primary reference-based quality signal.

#### LLM Judge (Claude Haiku)
Prompts Claude to score the reply on 5 dimensions (1–10 each) and return structured JSON. Evaluates: relevance, tone, completeness, accuracy, clarity.

**Strength:** Closest proxy to a human quality review. Catches nuanced issues (wrong tone, missing next steps, false information).  
**Weakness:** Non-deterministic; can be biased toward verbose replies.  
**Mitigation:** Uses a cheaper, faster model (Haiku) to separate the judge from the generator (Sonnet), reducing self-serving bias.  
**Role here:** Highest-weight component (40%) because it most directly measures customer satisfaction.

### Composite Score Formula

```
composite = 0.40 × (LLM Judge avg / 10)
          + 0.35 × Semantic Similarity
          + 0.25 × ROUGE-L
```

**Weight rationale:**
- LLM Judge gets the highest weight (40%) because it captures the most human-relevant quality dimensions
- Semantic Similarity (35%) is more robust than ROUGE for free-form text
- ROUGE-L (25%) provides a low-cost consistency anchor
- Without a reference reply, composite = LLM Judge only (normalized to 0–1)

### Per-response output example

```
Metric                Score    Notes
─────────────────────────────────────────────────────
  Relevance           8.000    Addresses the sync issue directly
  Tone                9.000    Empathetic and professional
  Completeness        7.000    Provides steps but no ETA
  Accuracy            8.000    Steps are correct
  Clarity             9.000    Easy to follow
LLM Judge (avg)       8.200    Mean of 5 dimensions
ROUGE-1               0.312    Unigram overlap w/ reference
ROUGE-2               0.128    Bigram overlap w/ reference
ROUGE-L               0.289    Longest common subsequence
Semantic Similarity   0.731    Sentence-BERT cosine distance
─────────────────────────────────────────────────────
Composite Score       0.680    Weighted blend
```

### Overall system score

After evaluating all test emails, the runner prints mean/min/max for each metric across the batch — giving an aggregate system quality score.

### Validating that the metric reflects real quality

The LLM judge is independently prompted using a *different model* (Haiku) than the generator (Sonnet). The judge prompt instructs Claude to be critical and reserve high scores for genuinely excellent replies.

Sanity-check: run `python main.py --id 20` (a complaint email). Compare the score for the generated reply vs. a deliberately weak reply like "Thanks for contacting us." The gap in composite score validates that the metric discriminates real quality.

---

## How I used AI tools

- **Claude Sonnet 4.6** — email reply generation (the generator itself)
- **Claude Haiku 4.5** — LLM-as-judge evaluation (fast, cheap, separate from generator)
- **Claude Code** — assisted with code scaffolding, README drafting, and reviewing the evaluation design
- The dataset was hand-reviewed and edited for quality after initial drafting with Claude Code

---

## Requirements

```
anthropic>=0.40.0
rouge-score>=0.1.2
sentence-transformers>=3.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
rich>=13.0.0
python-dotenv>=1.0.0
```

Python 3.10+ required (uses `str | None` union syntax).
