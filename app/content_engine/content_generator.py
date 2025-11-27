"""
app/content_engine/content_generator.py
Supports:
  1) Groq (LLaMA-based)
  2) Google Generative AI (Gemini / Palm)
  3) Local fallback if both fail
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ==========================
# LLM CLIENTS
# ==========================

# Groq client (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# Google Generative AI client (optional)
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except Exception:
    genai = None
    GOOGLE_AVAILABLE = False

# Dynamic prompt builder
from .dynamic_prompt import generate_engaging_prompt

# Trend optimizer
from app.integrations.trend_fetcher import TrendFetcher
from app.content_engine.trend_based_optimizer import TrendBasedOptimizer

# Sheets logging
from app.integrations.sheets_connector import append_row

# Optional tools
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

try:
    import language_tool_python
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


# ==========================
# LOCAL FALLBACK
# ==========================

def _local_generate(prompt: str, n: int = 3) -> List[str]:
    """
    Simple local generator used if all LLMs fail.
    """
    return [f"{prompt}\n\n[LOCAL VARIANT {i+1}]" for i in range(n)]


# ==========================
# GROQ CALL
# ==========================

def _call_groq(prompt: str, model: Optional[str] = None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
    )
    return resp.choices[0].message.content


# ==========================
# GOOGLE (GEMINI / GENERATIVE AI) CALL
# ==========================

def _call_google(prompt: str, model: Optional[str] = None) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai SDK is not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # Configure SDK
    genai.configure(api_key=api_key)

    # IMPORTANT: you MUST put the EXACT model name that works for your key in .env
    # Example values (depends on your Google account!):
    #  - gemini-1.5-flash
    #  - gemini-1.5-pro
    #  - models/gemini-1.5-flash
    # Get the exact name from AI Studio "View Code" → Python snippet.
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    logger.info(f"Using Google model: {model_name}")
    gmodel = genai.GenerativeModel(model_name)
    response = gmodel.generate_content(prompt)

    return getattr(response, "text", str(response))


# ==========================
# MAIN LLM GENERATION
# ==========================

def generate_variations(prompt: str, n: int = 2) -> List[str]:
    """
    Order:
      1) Groq (if GROQ_API_KEY present)
      2) Google Generative AI (if GEMINI_API_KEY present)
      3) Local fallback
    """
    logger.info(
        f"generate_variations: "
        f"GROQ_AVAILABLE={GROQ_AVAILABLE}, GOOGLE_AVAILABLE={GOOGLE_AVAILABLE}, "
        f"has_groq_key={bool(os.getenv('GROQ_API_KEY'))}, "
        f"has_google_key={bool(os.getenv('GEMINI_API_KEY'))}"
    )

    # 1) Try Groq
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            logger.info("🔹 Using Groq for content generation...")
            return [_call_groq(prompt) for _ in range(n)]
        except Exception:
            logger.exception("Groq failed → trying Google Generative AI...")

    # 2) Try Google Generative AI
    if GOOGLE_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            logger.info("🔹 Using Google Generative AI for content generation...")
            return [_call_google(prompt) for _ in range(n)]
        except Exception:
            logger.exception("Google Generative AI failed → using local fallback...")

    # 3) Fallback local
    logger.warning("⚠️ No LLM available → using LOCAL fallback generator.")
    return _local_generate(prompt, n)


# ==========================
# QUALITY SCORING
# ==========================

def score_quality(text: str) -> Dict:
    readability_score = None
    grammar_issues = None

    if TEXTSTAT_AVAILABLE:
        try:
            readability_score = textstat.flesch_reading_ease(text)
        except Exception:
            readability_score = None

    if LT_AVAILABLE:
        try:
            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(text)
            grammar_issues = len(matches)
        except Exception:
            grammar_issues = None

    return {
        "readability_score": readability_score,
        "grammar_issues": grammar_issues,
    }


# ==========================
# HASHTAG UTILITIES
# ==========================

def clean_punctuation_hashtags(text: str) -> str:
    words = text.split()
    cleaned = []
    for w in words:
        if w.startswith("#"):
            w = w.rstrip(",.?!;:")
        cleaned.append(w)
    return " ".join(cleaned)


def dedupe_hashtags(text: str) -> str:
    seen = set()
    out = []
    for w in text.split():
        if w.startswith("#"):
            key = w.lower()
            if key not in seen:
                seen.add(key)
                out.append(w)
        else:
            out.append(w)
    return " ".join(out)


def move_hashtags_to_end(text: str) -> str:
    words = text.split()
    tags = [w for w in words if w.startswith("#")]
    others = [w for w in words if not w.startswith("#")]
    return " ".join(others + tags)


def clean_and_order_hashtags(text: str) -> str:
    text = clean_punctuation_hashtags(text)
    text = dedupe_hashtags(text)
    text = move_hashtags_to_end(text)
    return text


# ==========================
# ENGAGEMENT RANKING
# ==========================

def optimize_with_engagement(
    candidates: List[Dict], past_metrics: Optional[Dict] = None
) -> List[Dict]:
    top_keywords = []
    if past_metrics:
        top_keywords = list(past_metrics.get("top_keywords", []))[:3]

    scored: List[Dict] = []

    for c in candidates:
        text = c.get("optimized_text", "")
        score = 0.0

        q = score_quality(text)
        if q["readability_score"] is not None:
            score += q["readability_score"] / 100.0
        if q["grammar_issues"] is not None:
            score -= min(1.0, 0.1 * q["grammar_issues"])

        for kw in top_keywords:
            if kw.lower() in text.lower():
                score += 0.2

        c["engagement_score"] = score
        scored.append((score, c))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_sorted]


# ==========================
# PIPELINE
# ==========================

def generate_final_variations(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    n: int = 2,
    word_count: int = 50,
    past_metrics: Optional[Dict] = None,
) -> List[Dict]:

    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    tf = TrendFetcher()
    try:
        real_trends = tf.fetch_google_global_trends()
    except Exception as e:
        logger.warning(f"TrendFetcher failed, using empty trends: {e}")
        real_trends = []

    injected_keywords = keywords + [t for t in real_trends if t not in keywords]

    # Build a combined prompt using your dynamic prompt builder
    prompt = generate_engaging_prompt(
        topic,
        platform,
        injected_keywords,
        audience,
        tone,
        trends=real_trends,
        word_count=word_count,
    )

    raw_variants = generate_variations(prompt, n=n)

    optimizer = TrendBasedOptimizer()
    optimized_candidates: List[Dict] = []

    for text in raw_variants:
        opt = optimizer.run(text)

        cleaned = clean_and_order_hashtags(opt.get("optimized", text))
        opt["optimized_text"] = cleaned

        optimized_candidates.append(opt)

    final_order = optimize_with_engagement(optimized_candidates, past_metrics)

    results: List[Dict] = []
    for item in final_order:
        optimized_text = item.get("optimized_text", "")

        results.append(
            {
                "text": optimized_text,
                "quality": score_quality(optimized_text),
                "meta": {
                    "topic": topic,
                    "platform": platform,
                    "audience": audience,
                    "injected_keywords": injected_keywords,
                    "trend_score": item.get("trend_score", 0),
                    "trend_insights": item.get("insights", {}),
                },
            }
        )

        try:
            append_row(
                "generated_content",
                [
                    datetime.utcnow().isoformat(),
                    platform,
                    topic[:40] + "...",
                    optimized_text[:80] + "...",
                    item.get("trend_score", 0),
                ],
            )
        except Exception:
            pass

    return results
