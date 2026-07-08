"""
Email Response Generator
Uses TF-IDF retrieval to find similar past emails, then few-shot prompts an LLM
to generate a contextually grounded reply.

Supported providers:
  anthropic  — Claude Sonnet 4.6  (default)
  groq       — Llama 3.3 70B via Groq inference (fast, free tier available)
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "groq": "llama-3.3-70b-versatile",
}


class EmailResponseGenerator:
    def __init__(
        self,
        dataset_path: str = "data/emails.json",
        top_k: int = 3,
        provider: str = "anthropic",
        model: str | None = None,
    ):
        self.provider = provider
        self.top_k = top_k
        self.model = model or PROVIDER_DEFAULTS.get(provider)

        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        elif provider == "groq":
            from groq import Groq
            self._client = Groq()
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose 'anthropic' or 'groq'."
            )

        with open(dataset_path) as f:
            self.dataset = json.load(f)

        corpus = [f"{e['subject']} {e['incoming_email']}" for e in self.dataset]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------ #

    def _retrieve_similar(self, query: str) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(scores)[-self.top_k :][::-1]
        return [self.dataset[i] for i in top_indices]

    def _call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=messages,
            )
            return response.content[0].text.strip()

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=600,
                messages=messages,
            )
            return response.choices[0].message.content.strip()

        raise RuntimeError(f"Provider '{self.provider}' not handled in _call_llm")

    # ------------------------------------------------------------------ #

    def generate(self, incoming_email: str, subject: str = "") -> dict:
        query = f"{subject} {incoming_email}".strip()
        examples = self._retrieve_similar(query)

        few_shot_block = ""
        for ex in examples:
            few_shot_block += (
                f"\n---\nSubject: {ex['subject']}\n"
                f"Customer Email:\n{ex['incoming_email']}\n\n"
                f"Reply:\n{ex['reply']}\n"
            )

        prompt = f"""You are a professional customer support agent for Hiver, a B2B SaaS product that gives teams shared email inboxes inside Gmail.

Below are {self.top_k} examples of real past email exchanges and the replies that were sent. Study their tone, structure, and level of detail.

{few_shot_block}
---

Now write a reply to the following new customer email. Match the tone and style of the examples: concise, empathetic, solution-focused, and professional. Do not start with "I hope this email finds you well" or generic openers.

Subject: {subject}
Customer Email:
{incoming_email}

Reply:"""

        generated_text = self._call_llm(prompt)

        return {
            "generated_reply": generated_text,
            "retrieved_examples": [
                {"id": e["id"], "subject": e["subject"], "category": e["category"]}
                for e in examples
            ],
            "provider": self.provider,
            "model": self.model,
            "retrieval_method": "tfidf_cosine",
        }
