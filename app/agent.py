"""Groq-powered conversational SHL assessment recommender agent (RAG-based)."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("shl_agent")

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
GROQ_MODEL = "llama-3.3-70b-versatile"
RAG_TOP_K = 50  # candidates passed to the LLM


def _first_type(raw: str) -> str:
    """Return the first single-letter type code from a ';'-separated string."""
    return raw.split(";")[0].strip() if raw else "K"


class SHLAgent:
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required")

        self._client = Groq(api_key=api_key)

        with open(CATALOG_PATH) as fh:
            self.catalog: List[Dict] = json.load(fh)

        # Lookup indexes
        self._url_index: Dict[str, Dict] = {item["url"]: item for item in self.catalog}
        self._norm_index: Dict[str, Dict] = {
            item["url"].rstrip("/").lower(): item for item in self.catalog
        }
        self._slug_index: Dict[str, Dict] = {
            item["url"].rstrip("/").split("/")[-1].lower(): item
            for item in self.catalog
        }

        # TF-IDF index for retrieval
        texts = [
            f'{item["name"]} {item.get("test_type","")}'.lower()
            for item in self.catalog
        ]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

        logger.info("SHLAgent: loaded %d catalog items", len(self.catalog))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, k: int = RAG_TOP_K) -> List[Dict]:
        """Return top-k catalog items most similar to query via TF-IDF."""
        q_vec = self._vectorizer.transform([query.lower()])
        sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.catalog[i] for i in top_idx]

    def _retrieve_by_names(self, names: List[str]) -> List[Dict]:
        """Retrieve catalog items that closely match a list of names."""
        results = []
        name_lower = [n.lower() for n in names]
        for item in self.catalog:
            iname = item["name"].lower()
            if any(n in iname or iname in n for n in name_lower):
                results.append(item)
        return results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _format_candidates(items: List[Dict]) -> str:
        lines = []
        for item in items:
            tt = item.get("test_type", "K")
            remote = "Y" if item.get("remote_support") == "Yes" else "N"
            adaptive = "Y" if item.get("adaptive_support") == "Yes" else "N"
            # Use URL slug as short ID; full URL also included
            lines.append(
                f'- {item["name"]} | {item["url"]} | {tt} | R:{remote} A:{adaptive}'
            )
        return "\n".join(lines)

    def _build_system_prompt(self, candidates: List[Dict]) -> str:
        catalog_block = self._format_candidates(candidates)
        return f"""You are an SHL Assessment Recommender helping hiring managers select assessments for job roles.

RETRIEVED CATALOG SUBSET ({len(candidates)} assessments relevant to this conversation):
{catalog_block}

TYPE CODES: A=Ability  B=Biodata  C=Competency  D=Development  E=Engagement  K=Knowledge/Skills  P=Personality  S=Situational

BEHAVIORAL RULES:
1. CLARIFY: Only ask clarifying questions when the query has NO role and NO skill/context at all (e.g., "I need an assessment" with nothing else). If the user mentions a job title, skill, or description — that is enough to recommend. Do NOT over-ask.
2. RECOMMEND: As soon as you know the role or key skill (e.g., "Java developer", "sales rep", "COO"), recommend 1-10 relevant assessments from the catalog subset. Job descriptions are sufficient — recommend directly.
3. REFINE: If the user changes constraints (add personality tests, remove coding tests, duration limit), update the shortlist accordingly.
4. COMPARE: If the user asks to compare assessments, answer using only catalog data (name, types, remote/adaptive).
5. REFUSE: Politely decline off-topic requests (salary, legal, general HR). Stay focused on SHL assessments.
6. GROUNDED: Names and URLs MUST be copied verbatim from the catalog subset above. Never fabricate.

RESPONSE — pure JSON only (no markdown, no extra text):
{{
  "reply": "<your conversational response>",
  "recommendations": [],
  "end_of_conversation": false
}}

When recommending (1-10 items):
{{
  "reply": "Here are N assessments for [role]...",
  "recommendations": [
    {{"name": "<exact name>", "url": "<exact url>", "test_type": "<first letter>"}}
  ],
  "end_of_conversation": false
}}

RULES:
- "recommendations" MUST be [] when clarifying or refusing.
- "recommendations" has 1-10 items only when committed to a shortlist.
- "end_of_conversation" is true only when the user is done/satisfied.
- "test_type" = first letter of the catalog entry's type string.
- Output ONLY valid JSON. No markdown fences."""

    # ------------------------------------------------------------------
    # URL validation
    # ------------------------------------------------------------------

    def _resolve_item(self, url: str) -> Optional[Dict]:
        item = self._url_index.get(url)
        if item:
            return item
        item = self._norm_index.get(url.rstrip("/").lower())
        if item:
            return item
        slug = url.rstrip("/").split("/")[-1].lower()
        return self._slug_index.get(slug)

    # ------------------------------------------------------------------
    # Main chat method
    # ------------------------------------------------------------------

    def chat(self, messages: List[Dict]) -> Dict:
        """
        Process a stateless conversation history and return the agent reply.

        Args:
            messages: list of {"role": "user"|"assistant", "content": str}

        Returns:
            dict with keys: reply (str), recommendations (list), end_of_conversation (bool)
        """
        # Build a retrieval query from the full conversation
        conv_text = " ".join(m["content"] for m in messages)

        # Retrieve relevant candidates
        candidates = self._retrieve(conv_text, k=RAG_TOP_K)

        # For compare queries, also ensure named assessments are included
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if re.search(r"\bcompar\b|\bdifference\b|\bvs\b", last_user, re.I):
            # Extract potential assessment names and add them to candidates
            extra = self._retrieve_by_names(last_user.split())
            seen = {c["url"] for c in candidates}
            for item in extra:
                if item["url"] not in seen:
                    candidates.append(item)
                    seen.add(item["url"])

        # Build system prompt with candidates
        system_prompt = self._build_system_prompt(candidates)

        # Build Groq message list
        groq_msgs: List[Dict] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            groq_msgs.append({"role": role, "content": m["content"]})

        try:
            completion = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=groq_msgs,
                temperature=0.1,
                max_tokens=1024,
            )
            raw = completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            return self._error_response()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s | raw: %.300s", exc, raw)
            match = re.search(r'"reply"\s*:\s*"([^"]+)"', raw)
            reply = match.group(1) if match else "I encountered a parsing issue. Please rephrase."
            return {"reply": reply, "recommendations": [], "end_of_conversation": False}

        # Validate recommendations against catalog
        clean_recs: List[Dict] = []
        for r in parsed.get("recommendations", []):
            if not isinstance(r, dict):
                continue
            item = self._resolve_item(r.get("url", ""))
            if item is None:
                logger.warning("Rejected hallucinated URL: %s", r.get("url"))
                continue
            clean_recs.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": _first_type(r.get("test_type") or item.get("test_type", "K")),
            })
            if len(clean_recs) == 10:
                break

        return {
            "reply": str(parsed.get("reply", "")).strip(),
            "recommendations": clean_recs,
            "end_of_conversation": bool(parsed.get("end_of_conversation", False)),
        }

    @staticmethod
    def _error_response() -> Dict:
        return {
            "reply": "I'm having trouble processing your request. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }
