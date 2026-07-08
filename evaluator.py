"""
Email Evaluation System
Multi-metric scorer: ROUGE + Semantic Similarity + LLM-as-Judge → composite score.

Why these metrics?
- ROUGE: fast lexical overlap baseline; catches topic drift but blind to paraphrasing
- Semantic Similarity: cosine distance between sentence embeddings; captures meaning
  even when wording differs — essential for free-form email
- LLM Judge: holistic quality across 5 customer-care dimensions; closest proxy
  to a human quality review
- Composite: weighted blend (LLM 40 % + Semantic 35 % + ROUGE-L 25 %)
  because LLM judge is most aligned with customer satisfaction, semantic
  similarity is more robust than ROUGE, and ROUGE adds a low-cost consistency check
"""

import json
import re
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_EMBED_MODEL = None  # lazy-loaded
_CLIENT = Anthropic()

WEIGHTS = {"llm_judge": 0.40, "semantic_similarity": 0.35, "rougeL": 0.25}


def _embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #

def rouge_scores(generated: str, reference: str) -> dict:
    scores = _ROUGE.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def semantic_similarity(generated: str, reference: str) -> float:
    model = _embed_model()
    vecs = model.encode([generated, reference])
    sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    return round(float(sim), 4)


def llm_judge(
    incoming_email: str,
    generated_reply: str,
    reference_reply: str | None = None,
) -> dict:
    ref_section = (
        f"\n\nReference (human) Reply:\n{reference_reply}" if reference_reply else ""
    )

    prompt = f"""You are an expert evaluator of customer support email replies. Score the Generated Reply below on five dimensions, each on a scale of 1–10. Be critical and honest; reserve 9–10 for genuinely excellent replies.

Customer Email:
{incoming_email}

Generated Reply:
{generated_reply}{ref_section}

Scoring dimensions:
1. Relevance (1–10): Does the reply directly address the customer's specific issue?
2. Tone (1–10): Is it empathetic, professional, and appropriate for B2B support?
3. Completeness (1–10): Does it fully resolve the query or provide clear next steps?
4. Accuracy (1–10): Is the information given correct and trustworthy?
5. Clarity (1–10): Is the language clear, concise, and free of jargon?

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{
  "relevance":    {{"score": <1-10>, "reason": "<one sentence>"}},
  "tone":         {{"score": <1-10>, "reason": "<one sentence>"}},
  "completeness": {{"score": <1-10>, "reason": "<one sentence>"}},
  "accuracy":     {{"score": <1-10>, "reason": "<one sentence>"}},
  "clarity":      {{"score": <1-10>, "reason": "<one sentence>"}},
  "overall_comment": "<two sentences summarising quality>"
}}"""

    response = _CLIENT.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": "parse_failed", "raw": raw}


# --------------------------------------------------------------------------- #
# Composite evaluator
# --------------------------------------------------------------------------- #

def evaluate(
    incoming_email: str,
    generated_reply: str,
    reference_reply: str | None = None,
) -> dict:
    result: dict = {}

    # --- Lexical & semantic (only meaningful with a reference) ---
    if reference_reply:
        result["rouge"] = rouge_scores(generated_reply, reference_reply)
        result["semantic_similarity"] = semantic_similarity(
            generated_reply, reference_reply
        )

    # --- LLM judge (works with or without reference) ---
    result["llm_judge"] = llm_judge(incoming_email, generated_reply, reference_reply)

    # --- Composite score ---
    judge = result["llm_judge"]
    dims = ["relevance", "tone", "completeness", "accuracy", "clarity"]
    judge_scores = [judge.get(d, {}).get("score", 5) for d in dims if isinstance(judge.get(d), dict)]
    llm_norm = (np.mean(judge_scores) / 10.0) if judge_scores else 0.5

    if reference_reply:
        rouge_l = result["rouge"]["rougeL"]
        sem_sim = result["semantic_similarity"]
        composite = (
            WEIGHTS["llm_judge"] * llm_norm
            + WEIGHTS["semantic_similarity"] * sem_sim
            + WEIGHTS["rougeL"] * rouge_l
        )
    else:
        # Without reference, composite = LLM judge only
        composite = llm_norm

    result["llm_judge_avg"] = round(llm_norm * 10, 2)
    result["composite_score"] = round(composite, 4)
    return result
