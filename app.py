"""
╔═══════════════════════════════════════════════════════════════╗
║         PlagioScan - AI-Powered Plagiarism Checker           ║
║         Built with Python & Streamlit | Demo Version         ║
╚═══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import re
import pickle
import os
import time
from datetime import datetime
from typing import Any

# ── scikit-learn ──────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── difflib for sentence-level highlighting ───────────────────
import difflib

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PlagioScan – Plagiarism Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CUSTOM CSS
# ══════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root variables ── */
:root {
    --accent:    #6C63FF;
    --accent2:   #FF6584;
    --green:     #22c55e;
    --yellow:    #f59e0b;
    --red:       #ef4444;
    --card-bg:   rgba(255,255,255,0.04);
    --border:    rgba(255,255,255,0.09);
    --radius:    14px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #0d0d14;
    color: #e2e2ef;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #0d0d14 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Space Grotesk', sans-serif; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #6C63FF, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: #9ca3af;
    margin: 0;
    font-weight: 400;
}

/* ── AI Badge ── */
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(90deg, rgba(108,99,255,0.2), rgba(255,101,132,0.2));
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.5px;
    margin-bottom: 16px;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(108,99,255,0.4); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.82rem; color: #9ca3af; margin-top: 4px; font-weight: 500; letter-spacing: 0.4px; text-transform: uppercase; }

/* ── Section cards ── */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
}

/* ── Score ring / progress ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 28px;
}
.score-big {
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
}
.score-label { font-size: 0.9rem; color: #9ca3af; margin-top: 8px; }

/* ── Highlighted text ── */
.text-display {
    background: #12121f;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.75;
    max-height: 320px;
    overflow-y: auto;
}
.hl-red   { background: rgba(239,68,68,0.28);   border-radius: 4px; padding: 1px 3px; color: #fca5a5; }
.hl-yellow{ background: rgba(245,158,11,0.28);  border-radius: 4px; padding: 1px 3px; color: #fcd34d; }
.hl-clean { color: #d1fae5; }

/* ── Source cards ── */
.source-card {
    background: #12121f;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.source-title { font-weight: 600; font-size: 0.9rem; color: #e2e2ef; }
.source-link  { font-size: 0.75rem; color: #6C63FF; }
.source-pct   { font-size: 1.3rem; font-weight: 700; }

/* ── Report row ── */
.report-row {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.report-score-dot {
    width: 44px; height: 44px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem;
    flex-shrink: 0;
}

/* ── Sidebar nav ── */
.nav-item {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 4px;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.92rem;
    color: #9ca3af;
    transition: all 0.15s;
    display: flex; align-items: center; gap: 10px;
}
.nav-item:hover, .nav-item.active {
    background: rgba(108,99,255,0.15);
    color: #a78bfa;
}

/* ── Word count bar ── */
.wc-bar {
    display: flex; gap: 20px;
    font-size: 0.78rem; color: #9ca3af;
    padding: 8px 4px;
}
.wc-item span { color: #a78bfa; font-weight: 600; }

/* ── Progress bar override ── */
.stProgress > div > div { border-radius: 99px !important; }

/* ── Button override ── */
.stButton > button {
    background: linear-gradient(90deg, #6C63FF, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(108,99,255,0.35) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.72rem;
    color: #4b5563;
    padding: 24px 0 8px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── How-it-works step ── */
.hiw-step {
    display: flex; gap: 18px; align-items: flex-start;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    margin-bottom: 12px;
}
.hiw-num {
    background: linear-gradient(135deg, #6C63FF, #818cf8);
    border-radius: 50%;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
    flex-shrink: 0; color: white;
}

/* ── Compare result ── */
.compare-box {
    background: #12121f;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    min-height: 220px;
    font-size: 0.84rem;
    line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
    overflow-y: auto;
    max-height: 300px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SAMPLE DATABASE  (10-12 paragraphs across topics)
# ══════════════════════════════════════════════════════════════
SAMPLE_DB = {
    "education_1": """
    Education is the most powerful weapon which you can use to change the world. The purpose of
    education is not merely to fill a bucket but to light a fire. In India, education has been
    revered since ancient times through the Gurukul system where students lived with their
    teachers and received holistic learning encompassing spirituality, mathematics, and arts.
    Modern education must balance academic rigor with emotional intelligence, creativity, and
    critical thinking to prepare students for a rapidly changing world.
    """,
    "education_2": """
    Digital literacy has become as fundamental as reading and writing in the twenty-first century.
    Schools worldwide are integrating technology into classrooms, using interactive whiteboards,
    educational apps, and online learning platforms. The COVID-19 pandemic accelerated the shift
    toward remote learning, exposing both the potential and the inequalities of digital education.
    Students in rural areas often lack access to reliable internet, creating a digital divide
    that policymakers must urgently address through infrastructure investment.
    """,
    "technology_1": """
    Artificial intelligence is transforming every aspect of modern life, from healthcare diagnostics
    to autonomous vehicles and personalized content recommendations. Machine learning algorithms
    analyze vast datasets to identify patterns invisible to human observers. Deep learning neural
    networks, inspired by the human brain's architecture, achieve remarkable accuracy in image
    recognition, natural language processing, and predictive analytics. The ethical implications
    of AI decision-making in critical domains such as criminal justice and medical treatment
    demand careful governance frameworks and transparent algorithmic accountability.
    """,
    "technology_2": """
    Blockchain technology offers a decentralized and immutable ledger for recording transactions
    without the need for a central authority. Originally developed to support Bitcoin cryptocurrency,
    blockchain has found applications in supply chain management, voting systems, healthcare records,
    and smart contracts. The technology ensures transparency and security through cryptographic hashing.
    India's government has explored blockchain solutions for land registry, identity management,
    and reducing corruption in public service delivery systems.
    """,
    "environment_1": """
    Climate change represents the most urgent existential threat facing humanity in the twenty-first
    century. Rising greenhouse gas emissions from fossil fuel combustion, deforestation, and industrial
    agriculture are driving global temperatures to unprecedented levels. The Intergovernmental Panel
    on Climate Change warns that limiting warming to 1.5 degrees Celsius requires immediate and
    drastic reductions in carbon emissions across all sectors. Renewable energy technologies such
    as solar photovoltaics and offshore wind have achieved remarkable cost reductions, making
    clean energy economically competitive with coal and natural gas.
    """,
    "environment_2": """
    Biodiversity loss is occurring at a rate estimated to be one thousand times faster than natural
    background extinction rates, driven primarily by habitat destruction, pollution, invasive species,
    and climate change. Tropical rainforests, which harbor more than half of all terrestrial species,
    are being cleared at alarming rates for agriculture and timber. Marine ecosystems face threats
    from plastic pollution, ocean acidification, and overfishing. Conservation strategies must
    integrate community engagement, indigenous knowledge, and economic incentives to protect
    critical habitats and endangered species effectively.
    """,
    "indian_culture_1": """
    India's cultural heritage spans more than five thousand years, encompassing an extraordinary
    diversity of languages, religions, art forms, and philosophical traditions. The country is
    home to six major religions and hundreds of dialects, making it one of the most pluralistic
    civilizations in human history. Classical dance forms such as Bharatanatyam, Kathak, and
    Odissi embody centuries of devotional expression and technical mastery. Indian cuisine varies
    dramatically by region, reflecting local geography, agriculture, and cultural influences from
    Central Asian, Persian, and European traders and conquerors who shaped the subcontinent's history.
    """,
    "science_1": """
    Quantum computing harnesses the principles of quantum mechanics to perform computations that are
    fundamentally impossible for classical computers. Unlike classical bits which exist as either
    zero or one, quantum bits or qubits can exist in superposition, representing both states
    simultaneously. Quantum entanglement allows qubits to be correlated across distances, enabling
    parallel processing at an exponential scale. Pharmaceutical companies are using quantum simulations
    to model molecular interactions for drug discovery, potentially accelerating the development
    of treatments for cancer, Alzheimer's disease, and antibiotic-resistant infections.
    """,
    "science_2": """
    CRISPR-Cas9 gene editing technology has revolutionized biomedical research by enabling precise
    modification of DNA sequences in living organisms. Scientists can now correct genetic mutations
    responsible for hereditary diseases such as sickle cell anemia, cystic fibrosis, and Huntington's
    disease. Agricultural researchers apply CRISPR to develop drought-resistant crops and reduce
    the need for chemical pesticides. However, the prospect of germline editing that would pass
    genetic changes to future generations raises profound ethical questions about human enhancement,
    consent, and social equity that international regulatory bodies are actively debating.
    """,
    "economics_1": """
    India's economy has emerged as one of the fastest-growing major economies in the world,
    driven by a young demographic dividend, expanding middle class, and rapidly growing digital
    infrastructure. The Information Technology sector, centered in cities like Bengaluru, Hyderabad,
    and Pune, contributes significantly to export earnings and employment. Government initiatives
    such as Make in India, Startup India, and the Production Linked Incentive scheme aim to
    boost domestic manufacturing and reduce dependence on imports. However, income inequality,
    rural unemployment, and informal labor markets remain persistent structural challenges.
    """,
    "health_1": """
    Mental health awareness has grown significantly in recent years as societies recognize the
    profound impact of psychological well-being on productivity, relationships, and physical health.
    Depression and anxiety disorders affect hundreds of millions of people worldwide, yet stigma
    and limited access to professional care prevent many from seeking treatment. Mindfulness
    meditation, cognitive behavioral therapy, and community support networks have demonstrated
    effectiveness in managing common mental health conditions. Digital mental health platforms
    and teletherapy services are expanding access to psychological support, particularly in
    underserved rural and remote communities across developing nations.
    """,
}

# ── Fake matched sources pool ──────────────────────────────────
SOURCE_POOL = [
    {
        "title": "Wikipedia – Overview Article",
        "url": "https://en.wikipedia.org/wiki/sample",
        "icon": "🌐",
    },
    {
        "title": "ResearchGate – Academic Paper",
        "url": "https://www.researchgate.net/sample",
        "icon": "📄",
    },
    {
        "title": "National Geographic – Feature",
        "url": "https://www.nationalgeographic.com/sample",
        "icon": "🌍",
    },
    {
        "title": "IEEE Xplore – Technical Journal",
        "url": "https://ieeexplore.ieee.org/sample",
        "icon": "🔬",
    },
    {
        "title": "JSTOR – Scholarly Article",
        "url": "https://www.jstor.org/sample",
        "icon": "📚",
    },
    {
        "title": "Britannica – Encyclopedia Entry",
        "url": "https://www.britannica.com/sample",
        "icon": "📖",
    },
    {
        "title": "MIT OpenCourseWare – Lecture Notes",
        "url": "https://ocw.mit.edu/sample",
        "icon": "🎓",
    },
    {
        "title": "PubMed – Biomedical Literature",
        "url": "https://pubmed.ncbi.nlm.nih.gov/sample",
        "icon": "🏥",
    },
]

# ══════════════════════════════════════════════════════════════
#  NLTK SETUP (Stopwords & Lemmatization)
# ══════════════════════════════════════════════════════════════
STOPWORDS: set[str] = set()
LEMMATIZER: Any = None
NLTK_AVAILABLE: bool = False

try:  # type: ignore
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore

    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")  # type: ignore
    except LookupError:
        nltk.download("punkt", quiet=True)  # type: ignore
    try:
        nltk.data.find("corpora/stopwords")  # type: ignore
    except LookupError:
        nltk.download("stopwords", quiet=True)  # type: ignore
    try:
        nltk.data.find("corpora/wordnet")  # type: ignore
    except LookupError:
        nltk.download("wordnet", quiet=True)  # type: ignore
    try:
        nltk.data.find("corpora/omw-1.4")  # type: ignore
    except LookupError:
        nltk.download("omw-1.4", quiet=True)  # type: ignore

    # Reassign globals with proper type ignores
    STOPWORDS = set(stopwords.words("english"))  # type: ignore  # noqa: F811
    LEMMATIZER = WordNetLemmatizer()  # type: ignore  # noqa: F811
    NLTK_AVAILABLE = True  # type: ignore  # noqa: F811
except ImportError:
    # Keep initialized defaults
    pass


# ══════════════════════════════════════════════════════════════
#  TEXT PROCESSING UTILITIES (ADVANCED)
# ══════════════════════════════════════════════════════════════


def preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Advanced text preprocessing:
    - Convert to lowercase
    - Remove punctuation
    - Tokenize words
    - Remove stopwords (optional)
    - Apply lemmatization (optional)
    - Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not NLTK_AVAILABLE:
        return text

    # Tokenize
    if NLTK_AVAILABLE:
        try:
            from nltk.tokenize import word_tokenize as wt  # type: ignore

            tokens = wt(text)  # type: ignore
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    # Remove stopwords
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]

    # Lemmatize
    if lemmatize and LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]

    return " ".join(tokens)


def generate_ngrams(text: str, n: int = 2) -> list[str]:
    """
    Generate n-grams from text.
    n=2 for bigrams, n=3 for trigrams, etc.
    """
    words: list[str] = text.split()
    ngrams: list[str] = []
    for i in range(len(words) - n + 1):
        ngram: str = " ".join(words[i : i + n])
        ngrams.append(ngram)
    return ngrams


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def char_count(text: str) -> int:
    return len(text)


def sentence_tokenize(text: str):
    """Simple sentence splitter using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compute_similarity(
    text1: str,
    text2: str,
    use_ngrams: bool = True,
    ngram_size: int = 2,
) -> float:
    """
    Advanced cosine similarity using TF-IDF with optional n-grams.

    Args:
        text1: First text
        text2: Second text
        use_ngrams: Whether to use n-grams (default True for better semantic detection)
        ngram_size: Size of n-grams (2 for bigrams, 3 for trigrams)

    Returns:
        Similarity score between 0 and 1
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        # Preprocess both texts
        prep1 = preprocess(text1)
        prep2 = preprocess(text2)

        if not prep1 or not prep2:
            return 0.0

        # Configure vectorizer with n-grams if enabled
        if use_ngrams:
            vec = TfidfVectorizer(
                ngram_range=(1, ngram_size),
                max_features=500,
                lowercase=False,
            )
        else:
            vec = TfidfVectorizer(lowercase=False)

        # Fit and transform
        tfidf = vec.fit_transform([prep1, prep2])  # type: ignore
        # Convert sparse matrix to dense array
        tfidf_dense: np.ndarray[Any, Any] = np.asarray(tfidf.todense())  # type: ignore

        # Compute cosine similarity
        score = cosine_similarity(tfidf_dense[0:1], tfidf_dense[1:2])[0][0]
        return float(score)
    except Exception:
        return 0.0


def apply_sensitivity_factor(
    score: float,
    sensitivity: str,
) -> float:
    """
    Adjust similarity score based on sensitivity level.

    Args:
        score: Original similarity score (0-1)
        sensitivity: "Low", "Medium", or "High"

    Returns:
        Adjusted score
    """
    if sensitivity == "Low":
        # Strict: require higher similarity (penalize borderline matches)
        return score**1.3 if score > 0 else 0
    elif sensitivity == "High":
        # Lenient: detect even small similarities (boost borderline matches)
        return min(1.0, score**0.7)
    else:  # Medium (default)
        # Balanced
        return score


def check_plagiarism(
    input_text: str,
    sensitivity: str = "Medium",
) -> tuple[float, list[float], list[dict[str, Any]]]:
    """
    Advanced plagiarism detection against sample database.

    Args:
        input_text: Text to check
        sensitivity: Detection sensitivity ("Low", "Medium", or "High")

    Returns:
        Tuple of (overall_percentage, per_sentence_scores, matched_sources)
    """
    sentences = sentence_tokenize(input_text)
    if not sentences:
        return 0.0, [], []

    # ── Per-sentence similarity (using n-grams for better detection) ───
    sentence_scores: list[float] = []
    for sent in sentences:
        max_sim: float = 0.0
        for key, sample in SAMPLE_DB.items():
            # Use bigrams for semantic similarity
            sim = compute_similarity(sent, sample, use_ngrams=True, ngram_size=2)
            sim = apply_sensitivity_factor(sim, sensitivity)
            if sim > max_sim:
                max_sim = sim
        sentence_scores.append(max_sim)

    # ── Overall score against combined database ────────────────────────
    combined_db = " ".join(SAMPLE_DB.values())
    overall = compute_similarity(input_text, combined_db, use_ngrams=True, ngram_size=2)
    overall = apply_sensitivity_factor(overall, sensitivity)

    # ── Average sentence score ─────────────────────────────────────────
    avg_sent: float = float(np.mean(sentence_scores)) if sentence_scores else 0.0

    # ── Blended scoring (weighted average) ──────────────────────────────
    # 60% full-text similarity, 40% average sentence similarity
    blended: float = 0.6 * overall + 0.4 * avg_sent

    # ── Scale to appropriate percentage range ──────────────────────────
    # Map [0, 1] to percentage with adjusted ranges based on sensitivity
    if sensitivity == "Low":
        # Strict: max out at 70% unless > 0.8
        blended_pct = min(70.0, blended * 100) if blended < 0.8 else blended * 100
    elif sensitivity == "High":
        # Lenient: more aggressive scaling
        blended_pct = min(95.0, max(blended * 100, blended**0.5 * 100))
    else:  # Medium
        # Balanced: standard scaling
        blended_pct = blended * 100

    # Clamp to 0-100
    blended_pct = max(0.0, min(100.0, blended_pct))

    # ── Select top matched sources ────────────────────────────────────
    source_sims: list[tuple[str, float]] = []
    for key, sample in SAMPLE_DB.items():
        sim = compute_similarity(input_text, sample, use_ngrams=True, ngram_size=2)
        sim = apply_sensitivity_factor(sim, sensitivity)
        source_sims.append((key, sim))
    source_sims.sort(key=lambda x: x[1], reverse=True)

    # ── Compile matched sources ────────────────────────────────────────
    matched_sources: list[dict[str, Any]] = []
    threshold = (
        0.01 if sensitivity == "High" else 0.05 if sensitivity == "Low" else 0.02
    )

    for i, (key, sim) in enumerate(source_sims[:6]):
        if sim < threshold:
            continue
        pool_item = SOURCE_POOL[i % len(SOURCE_POOL)]
        topic = str(key.split("_")[0].capitalize())
        sim_pct = min(95.0, sim * 100)  # Cap individual source at 95%
        matched_sources.append(
            {
                "title": f"{pool_item['icon']} {pool_item['title']} ({topic})",
                "url": pool_item["url"],
                "pct": round(float(sim_pct), 1),
            }
        )

    return round(float(blended_pct), 1), sentence_scores, matched_sources


def highlight_text(
    sentences: list[str],
    scores: list[float],
    threshold_red: float = 0.5,
    threshold_yellow: float = 0.2,
) -> str:
    """
    Return HTML with highlighted sentences based on similarity scores.
    Scores can be in 0-1 range (normalized) or 0-100 range (percentage).
    """
    # Normalize scores to 0-1 if they appear to be percentages
    max_score = max(scores) if scores else 1.0
    if max_score > 1.5:  # Likely percentages, normalize
        normalized_scores = [s / 100.0 for s in scores]
    else:
        normalized_scores = scores

    parts: list[str] = []
    for sent, score in zip(sentences, normalized_scores):
        escaped: str = (
            sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if score >= threshold_red:
            parts.append(
                f'<span class="hl-red" title="Similarity: {score:.0%}">{escaped}</span>'
            )
        elif score >= threshold_yellow:
            parts.append(
                f'<span class="hl-yellow" title="Similarity: {score:.0%}">{escaped}</span>'
            )
        else:
            parts.append(f'<span class="hl-clean">{escaped}</span>')
    return " ".join(parts)


def score_color(pct: float) -> str:
    if pct < 30:
        return "#22c55e"
    if pct < 60:
        return "#f59e0b"
    return "#ef4444"


def score_label(pct: float) -> str:
    if pct < 20:
        return "✅ Original"
    if pct < 30:
        return "🟢 Low Risk"
    if pct < 50:
        return "🟡 Moderate"
    if pct < 70:
        return "🟠 High Risk"
    return "🔴 Plagiarised"


def score_emoji(pct: float) -> str:
    if pct < 30:
        return "🟢"
    if pct < 60:
        return "🟡"
    return "🔴"


# ══════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════


def generate_pdf_report(
    text: str,
    pct: float,
    words: int,
    unique_words: int,
    plagiarised_words: int,
    sources: list[dict[str, Any]],
) -> bytes | None:
    """Generate a PDF report using fpdf2."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(
            0,
            14,
            "PlagioScan - Plagiarism Report",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 120, 140)
        pdf.cell(
            0,
            8,
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        pdf.ln(6)

        # Score box
        color = (
            (34, 197, 94) if pct < 30 else (245, 158, 11) if pct < 60 else (239, 68, 68)
        )
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 36)
        pdf.cell(
            0,
            22,
            f"{pct}%  Plagiarism Detected",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
            fill=True,
        )
        pdf.ln(4)

        # Stats
        pdf.set_text_color(40, 40, 60)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Document Statistics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"  Total Words: {words}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"  Unique Words: {unique_words}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0,
            7,
            f"  Potentially Plagiarised Words: {plagiarised_words}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(4)

        # Sources
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Matched Sources", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for s in sources:
            clean_title = s["title"].encode("ascii", errors="ignore").decode()
            pdf.cell(
                0,
                7,
                f"  - {clean_title}  |  Similarity: {s['pct']}%",
                new_x="LMARGIN",
                new_y="NEXT",
            )
        pdf.ln(4)

        # Text snippet
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Input Text (first 800 chars)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        clean_text = text[:800].encode("ascii", errors="ignore").decode()
        pdf.multi_cell(0, 6, clean_text)
        pdf.ln(4)

        # Footer
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 170)
        pdf.cell(
            0,
            6,
            "This is a demo report generated by PlagioScan. Results are based on a hardcoded sample database.",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        pdf_content: bytes = pdf.output()
        if isinstance(pdf_content, str):
            pdf_content = pdf_content.encode("latin-1")
        return pdf_content
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════
#  SESSION STATE – REPORTS PERSISTENCE
# ══════════════════════════════════════════════════════════════

REPORTS_FILE = "plagioscan_reports.pkl"


def load_reports() -> list[dict[str, Any]]:
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return []


def save_reports(reports: list[dict[str, Any]]) -> None:
    with open(REPORTS_FILE, "wb") as f:
        pickle.dump(reports, f)


if "reports" not in st.session_state:
    st.session_state.reports = load_reports()

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.2rem;">🔍</div>
        <div style="font-size:1.25rem; font-weight:700; color:#a78bfa; font-family:'Space Grotesk',sans-serif;">PlagioScan</div>
        <div style="font-size:0.7rem; color:#6b7280; margin-top:2px; font-family:'Space Grotesk',sans-serif;">AI-Powered Checker</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08); margin:10px 0 18px;">
    """,
        unsafe_allow_html=True,
    )

    pages = [
        ("🏠 Home", "Home"),
        ("🔍 Check Plagiarism", "Check Plagiarism"),
        ("⚖️ Compare Texts", "Compare Texts"),
        ("📋 My Reports", "My Reports"),
        ("❓ How It Works", "How It Works"),
    ]

    for icon_label, key in pages:
        active_cls = "active" if st.session_state.page == icon_label else ""
        if st.button(icon_label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = icon_label

    st.markdown(
        "<hr style='border-color:rgba(255,255,255,0.08); margin:18px 0 12px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div style="font-size:0.72rem; color:#4b5563; text-align:center; font-family:'Space Grotesk',sans-serif; padding:0 8px;">
        📊 Reports saved: <b style="color:#a78bfa;">{len(st.session_state.reports)}/8</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

current_page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════
if current_page == "🏠 Home":
    st.markdown(
        """
    <div class="hero-banner">
        <div class="ai-badge">✦ Powered by AI (Demo)</div>
        <div class="hero-title">Detect Plagiarism Instantly</div>
        <p class="hero-sub">
            Paste your text or upload a document — get a detailed similarity report in seconds.<br>
            Built for students, educators, and researchers.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Stats row ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["12+", "TF-IDF", "6", "Free"],
        ["Sample Sources", "Algorithm", "Fake Sources Shown", "Always"],
    ):
        col.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Features grid ─────────────────────────────────────────
    st.markdown("### ✨ What You Can Do")
    f1, f2, f3 = st.columns(3)
    features = [
        (
            "🔍",
            "Check Plagiarism",
            "Paste or upload text and get an instant similarity score with highlighted matches.",
        ),
        (
            "⚖️",
            "Compare Two Texts",
            "Side-by-side comparison of two documents with similarity breakdown.",
        ),
        (
            "📋",
            "Save Reports",
            "Last 8 reports are auto-saved with date, score, and word counts.",
        ),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(
            f"""
        <div class="section-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700; font-size:1rem; color:#e2e2ef; margin-bottom:6px;">{title}</div>
            <div style="font-size:0.82rem; color:#9ca3af; line-height:1.5;">{desc}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    # ── Quick start ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    sample_text = list(SAMPLE_DB.values())[0].strip()[:300]
    if st.button("🔍 Try with Sample Text →", use_container_width=False):
        st.session_state["quick_text"] = sample_text
        st.session_state.page = "🔍 Check Plagiarism"
        st.rerun()

    st.markdown(
        f"""
    <div class="footer">
        🔒 PlagioScan – Demo Application &nbsp;|&nbsp; 
        Built with Python & Streamlit &nbsp;|&nbsp;
        This is a frontend-style demo using hardcoded sample paragraphs. Not a real plagiarism detection service.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
#  PAGE: CHECK PLAGIARISM
# ══════════════════════════════════════════════════════════════
elif current_page == "🔍 Check Plagiarism":
    st.markdown(
        """
    <div class="hero-banner" style="padding:28px 36px;">
        <div class="ai-badge">✦ Smart Analysis</div>
        <div class="hero-title" style="font-size:2rem;">Check for Plagiarism</div>
        <p class="hero-sub">Paste your text below or upload a .txt / .docx file</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Input area ────────────────────────────────────────────
    col_input, col_upload = st.columns([2, 1])

    with col_input:
        default_text = st.session_state.pop("quick_text", "")
        input_text: str = (
            st.text_area(  # type: ignore
                "📝 Enter or paste your text here",
                value=default_text,
                height=260,
                placeholder="Paste your essay, article, or any text here...",
                key="main_input",
            )
            or ""
        )
        # live word/char count
        wc = word_count(input_text)
        cc = char_count(input_text)
        sentences_count = (
            len(sentence_tokenize(input_text)) if input_text.strip() else 0
        )
        st.markdown(
            f"""
        <div class="wc-bar">
            <div>Words: <span>{wc}</span></div>
            <div>Characters: <span>{cc}</span></div>
            <div>Sentences: <span>{sentences_count}</span></div>
        </div>""",
            unsafe_allow_html=True,
        )

    with col_upload:
        st.markdown("**📎 Or Upload a File**")
        uploaded_file = st.file_uploader(
            "Supported: .txt, .docx",
            type=["txt", "docx"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            if uploaded_file.name.endswith(".txt"):
                input_text = uploaded_file.read().decode("utf-8", errors="ignore")
                st.success(f"✅ Loaded: {uploaded_file.name}")
            elif uploaded_file.name.endswith(".docx"):
                try:
                    import docx2txt  # type: ignore

                    input_text = docx2txt.process(uploaded_file)  # type: ignore
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                except ImportError:
                    try:
                        from docx import Document

                        doc = Document(uploaded_file)
                        input_text = "\n".join([p.text for p in doc.paragraphs])
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Could not read .docx: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Options
        st.markdown("**⚙️ Settings**")
        sensitivity = st.select_slider(
            "Sensitivity",
            options=["Low", "Medium", "High"],
            value="Medium",
            label_visibility="visible",
            help="Low: Strict exact matching | Medium: Balanced | High: Detect small similarities",
        )
        # Dynamic thresholds based on sensitivity
        if sensitivity == "Low":
            t_red, t_yellow = 0.6, 0.35
        elif sensitivity == "High":
            t_red, t_yellow = 0.35, 0.1
        else:  # Medium
            t_red, t_yellow = 0.5, 0.2

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Check button ──────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        check_btn = st.button("🔍 Analyse Text", use_container_width=True)

    if check_btn:
        if not input_text.strip():
            st.error("⚠️ Please enter some text or upload a file before checking.")
        elif wc < 10:
            st.warning("⚠️ Please enter at least 10 words for a meaningful analysis.")
        else:
            # ── Loading simulation ─────────────────────────
            prog_bar = st.progress(0, text="🔄 Initialising scanner…")
            time.sleep(0.4)
            prog_bar.progress(15, text="📊 Extracting features…")
            time.sleep(0.5)
            prog_bar.progress(35, text="🧮 Computing TF-IDF vectors…")
            time.sleep(0.6)
            prog_bar.progress(55, text="🔗 Matching against database…")
            time.sleep(0.5)
            prog_bar.progress(75, text="📌 Highlighting sentences…")
            time.sleep(0.5)

            # ── Run check ─────────────────────────────────
            # input_text is guaranteed to be str due to `or ""` fallback
            overall_pct, sent_scores, matched_sources = check_plagiarism(
                input_text, sensitivity=sensitivity
            )
            sentences = sentence_tokenize(input_text)

            prog_bar.progress(95, text="📝 Preparing report…")
            time.sleep(0.3)
            prog_bar.progress(100, text="✅ Complete!")
            time.sleep(0.3)
            prog_bar.empty()

            # ── Word stats ────────────────────────────────
            all_words: list[Any] = input_text.lower().split()
            unique_words: int = len(set(all_words))
            plag_word_estimate: int = int(wc * overall_pct / 100)

            # ── Save report ───────────────────────────────
            snippet: str = (
                input_text[:120] + "…" if len(input_text) > 120 else input_text
            )
            report: dict[str, Any] = {
                "date": datetime.now().strftime("%d %b %Y, %H:%M"),
                "score": overall_pct,
                "words": wc,
                "unique": unique_words,
                "text_snippet": snippet,
                "sources": matched_sources,
            }
            reports = st.session_state.reports
            reports.insert(0, report)
            st.session_state.reports = reports[:8]
            save_reports(st.session_state.reports)

            # ══════════════════════════════════════════════
            #  RESULTS
            # ══════════════════════════════════════════════
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # ── Score + breakdown ─────────────────────────
            r1, r2 = st.columns([1, 2])

            with r1:
                color = score_color(overall_pct)
                label = score_label(overall_pct)
                st.markdown(
                    f"""
                <div class="section-card score-ring-wrap">
                    <div style="font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:600;">Plagiarism Score</div>
                    <div class="score-big" style="color:{color};">{overall_pct}%</div>
                    <div style="font-size:1rem; margin-top:10px;">{label}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(int(overall_pct) / 100)

            with r2:
                st.markdown(
                    """<div class="section-title">Document Breakdown</div>""",
                    unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                for col, val, lbl in zip(
                    [m1, m2, m3],
                    [wc, unique_words, plag_word_estimate],
                    ["Total Words", "Unique Words", "⚠️ Plag. Words"],
                ):
                    col.markdown(
                        f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.7rem;">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                # sentence-level progress
                st.markdown("**Sentence-level Risk Distribution**")
                low = sum(1 for s in sent_scores if s < t_yellow)
                mid = sum(1 for s in sent_scores if t_yellow <= s < t_red)
                high = sum(1 for s in sent_scores if s >= t_red)
                total = len(sent_scores) or 1
                sb1, sb2, sb3 = st.columns(3)
                sb1.metric("🟢 Clean", low, delta=f"{low/total:.0%}")
                sb2.metric("🟡 Suspect", mid, delta=f"{mid/total:.0%}")
                sb3.metric("🔴 Matched", high, delta=f"{high/total:.0%}")

            # ── Highlighted text ──────────────────────────
            st.markdown("---")
            st.markdown("### 🖊️ Highlighted Text")
            st.markdown(
                """
            <div style="display:flex; gap:16px; margin-bottom:10px; font-size:0.78rem;">
                <span class="hl-red" style="padding:2px 8px;">🔴 High Match</span>
                <span class="hl-yellow" style="padding:2px 8px;">🟡 Moderate Match</span>
                <span class="hl-clean" style="padding:2px 8px;">🟢 Original</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            highlighted = highlight_text(sentences, sent_scores, t_red, t_yellow)
            st.markdown(
                f'<div class="text-display">{highlighted}</div>', unsafe_allow_html=True
            )

            # ── Matched sources ───────────────────────────
            st.markdown("---")
            st.markdown("### 🌐 Matched Sources")
            if matched_sources:
                for src in matched_sources:
                    s_color = score_color(src["pct"])
                    st.markdown(
                        f"""
                    <div class="source-card">
                        <div>
                            <div class="source-title">{src['title']}</div>
                            <div class="source-link">🔗 {src['url']}</div>
                        </div>
                        <div class="source-pct" style="color:{s_color};">{src['pct']}%</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No significant matches found in the sample database.")

            # ── PDF download ──────────────────────────────
            st.markdown("---")
            st.markdown("### 📥 Download Report")
            pdf_bytes = generate_pdf_report(
                input_text,
                overall_pct,
                wc,
                unique_words,
                plag_word_estimate,
                matched_sources,
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download Full PDF Report",
                    data=bytes(pdf_bytes),
                    file_name=f"PlagioScan_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.info("Install `fpdf2` to enable PDF download: `pip install fpdf2`")

# ══════════════════════════════════════════════════════════════
#  PAGE: COMPARE TWO TEXTS
# ══════════════════════════════════════════════════════════════
elif current_page == "⚖️ Compare Texts":
    st.markdown(
        """
    <div class="hero-banner" style="padding:28px 36px;">
        <div class="ai-badge">✦ Side-by-Side</div>
        <div class="hero-title" style="font-size:2rem;">Compare Two Documents</div>
        <p class="hero-sub">Paste two texts to see how similar they are, sentence by sentence.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📄 Document A")
        text_a = st.text_area(
            "Text A",
            height=220,
            placeholder="Paste first text here…",
            label_visibility="collapsed",
        )
        wca = word_count(text_a)
        st.markdown(
            f'<div class="wc-bar"><div>Words: <span>{wca}</span></div></div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("#### 📄 Document B")
        text_b = st.text_area(
            "Text B",
            height=220,
            placeholder="Paste second text here…",
            label_visibility="collapsed",
        )
        wcb = word_count(text_b)
        st.markdown(
            f'<div class="wc-bar"><div>Words: <span>{wcb}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    cmp_btn_col, _ = st.columns([1, 3])
    with cmp_btn_col:
        cmp_btn = st.button("⚖️ Compare Documents", use_container_width=True)

    if cmp_btn:
        if not text_a.strip() or not text_b.strip():
            st.error("⚠️ Please enter text in both fields.")
        else:
            prog = st.progress(0, text="Comparing…")
            for pct_val, msg in [
                (30, "Vectorising…"),
                (60, "Computing similarity…"),
                (90, "Analysing sentences…"),
                (100, "Done!"),
            ]:
                time.sleep(0.4)
                prog.progress(pct_val, text=msg)
            time.sleep(0.2)
            prog.empty()

            sim = compute_similarity(text_a, text_b)
            sim_pct = round(sim * 100, 1)
            color = score_color(sim_pct)
            label = score_label(sim_pct)

            st.markdown("---")
            st.markdown("### 📊 Comparison Results")

            sc_col, dn_col = st.columns([1, 3])
            with sc_col:
                st.markdown(
                    f"""
                <div class="section-card score-ring-wrap">
                    <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:600;">Similarity</div>
                    <div class="score-big" style="color:{color};">{sim_pct}%</div>
                    <div style="font-size:1rem; margin-top:10px;">{label}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(int(sim_pct) / 100)

            with dn_col:
                st.markdown("**Common / Different Words**")
                words_a = set(preprocess(text_a).split())
                words_b = set(preprocess(text_b).split())
                common = words_a & words_b
                only_a = words_a - words_b
                only_b = words_b - words_a
                d1, d2, d3 = st.columns(3)
                d1.metric("🔗 Common", len(common))
                d2.metric("🔵 Only in A", len(only_a))
                d3.metric("🟣 Only in B", len(only_b))

                st.markdown("<br>", unsafe_allow_html=True)
                # sentence diff
                sents_a = sentence_tokenize(text_a)
                sents_b = sentence_tokenize(text_b)
                matcher = difflib.SequenceMatcher(None, sents_a, sents_b)
                ratio = matcher.ratio()
                st.markdown(f"**Sequence Match Ratio:** `{ratio:.2%}`")

            # ── Diff view ─────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔬 Detailed Diff View")
            diff_a_html: list[str] = []
            diff_b_html: list[str] = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                for s in sents_a[i1:i2]:
                    e = (
                        s.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    if tag == "equal":
                        diff_a_html.append(f'<span class="hl-red">{e}</span>')
                    else:
                        diff_a_html.append(f'<span class="hl-clean">{e}</span>')
                for s in sents_b[j1:j2]:
                    e = (
                        s.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    if tag == "equal":
                        diff_b_html.append(f'<span class="hl-red">{e}</span>')
                    else:
                        diff_b_html.append(f'<span class="hl-clean">{e}</span>')

            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Document A** (🔴 = shared with B)")
                st.markdown(
                    f'<div class="compare-box">{" ".join(diff_a_html)}</div>',
                    unsafe_allow_html=True,
                )
            with cb:
                st.markdown("**Document B** (🔴 = shared with A)")
                st.markdown(
                    f'<div class="compare-box">{" ".join(diff_b_html)}</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════
#  PAGE: MY REPORTS
# ══════════════════════════════════════════════════════════════
elif current_page == "📋 My Reports":
    st.markdown(
        """
    <div class="hero-banner" style="padding:28px 36px;">
        <div class="ai-badge">✦ History</div>
        <div class="hero-title" style="font-size:2rem;">My Reports</div>
        <p class="hero-sub">Your last 8 plagiarism checks are stored here automatically.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    reports = st.session_state.reports
    if not reports:
        st.info("🗂️ No reports yet. Run a plagiarism check to see results here.")
    else:
        # Summary stats
        avg_score = np.mean([r["score"] for r in reports])
        max_score = max(r["score"] for r in reports)
        min_score = min(r["score"] for r in reports)

        s1, s2, s3, s4 = st.columns(4)
        for col, val, lbl in zip(
            [s1, s2, s3, s4],
            [
                len(reports),
                f"{avg_score:.1f}%",
                f"{max_score:.1f}%",
                f"{min_score:.1f}%",
            ],
            ["Total Reports", "Average Score", "Highest Score", "Lowest Score"],
        ):
            col.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.6rem;">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # clear button
        clear_col, _ = st.columns([1, 4])
        with clear_col:
            if st.button("🗑️ Clear All Reports"):
                st.session_state.reports = []
                save_reports([])
                st.rerun()

        st.markdown("---")

        for i, r in enumerate(reports):
            color = score_color(r["score"])
            emoji = score_emoji(r["score"])
            with st.expander(
                f"{emoji} Report #{i+1}  —  {r['date']}  |  Score: {r['score']}%"
            ):
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Plagiarism Score", f"{r['score']}%")
                rc2.metric("Total Words", r.get("words", "—"))
                rc3.metric("Unique Words", r.get("unique", "—"))
                st.markdown(f"**Text snippet:** _{r.get('text_snippet', 'N/A')}_")
                if r.get("sources"):
                    st.markdown("**Matched Sources:**")
                    for s in r["sources"]:
                        st.markdown(f"- {s['title']}  →  **{s['pct']}%**")

# ══════════════════════════════════════════════════════════════
#  PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif current_page == "❓ How It Works":
    st.markdown(
        """
    <div class="hero-banner" style="padding:28px 36px;">
        <div class="ai-badge">✦ Under the Hood</div>
        <div class="hero-title" style="font-size:2rem;">How It Works</div>
        <p class="hero-sub">A transparent look at PlagioScan's detection methodology.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "1",
            "Text Preprocessing",
            "Input text is lowercased, punctuation removed, and tokens extracted. Stopwords are filtered using NLTK, and lemmatization is applied to normalize word forms (e.g., 'running' → 'run'). This ensures fair semantic comparison.",
        ),
        (
            "2",
            "Sentence Tokenisation",
            "The text is split into individual sentences using regex-based splitting on punctuation boundaries (.!?) for granular similarity analysis.",
        ),
        (
            "3",
            "N-gram Vectorization",
            "Bigrams (2-word sequences) and unigrams (single words) are extracted and converted into TF-IDF vectors using scikit-learn. This captures both word importance and local context, improving detection of paraphrased content.",
        ),
        (
            "4",
            "Cosine Similarity",
            "The dot product of TF-IDF vectors divided by their magnitudes computes cosine similarity (0 = no match, 1 = identical). Each sentence and the full text are compared against all database samples.",
        ),
        (
            "5",
            "Sensitivity Adjustment",
            "Similarity scores are adjusted based on user-selected sensitivity: Low (strict matching), Medium (balanced), or High (detect paraphrases). This dynamically rescales thresholds for better accuracy.",
        ),
        (
            "6",
            "Blended Scoring & Reporting",
            "The final plagiarism % is a weighted blend: 60% full-text similarity + 40% average sentence similarity. Sentences are highlighted (🔴 ≥ 50%, 🟡 20-50%, 🟢 < 20%) and matched sources are ranked by similarity.",
        ),
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
        <div class="hiw-step">
            <div class="hiw-num">{num}</div>
            <div>
                <div style="font-weight:700; font-size:0.95rem; color:#e2e2ef; margin-bottom:4px;">{title}</div>
                <div style="font-size:0.85rem; color:#9ca3af; line-height:1.6;">{desc}</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📚 Sample Database Topics")
    topics = [
        ("🎓", "Education", "Modern education, digital literacy, Gurukul system"),
        ("💻", "Technology", "Artificial Intelligence, Blockchain, Quantum Computing"),
        ("🌿", "Environment", "Climate change, biodiversity, renewable energy"),
        ("🪔", "Indian Culture", "Heritage, classical dance, culinary diversity"),
        ("🔬", "Science", "Gene editing (CRISPR), quantum mechanics"),
        ("💰", "Economics", "Indian economy, IT sector, Make in India"),
        ("🧠", "Mental Health", "Depression, therapy, digital mental health"),
    ]
    tc = st.columns(4)
    for i, (icon, topic, desc) in enumerate(topics):
        tc[i % 4].markdown(
            f"""
        <div class="section-card" style="padding:14px 16px; margin-bottom:10px;">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-weight:600; font-size:0.88rem; color:#e2e2ef; margin:4px 0 2px;">{topic}</div>
            <div style="font-size:0.75rem; color:#9ca3af;">{desc}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        ⚠️ <b>Disclaimer:</b> PlagioScan is a demo application built for educational and portfolio purposes.<br>
        Results are based on a hardcoded sample database and do not represent real-world plagiarism detection.<br><br>
        This is a frontend-style demo using hardcoded samples · Built with Python & Streamlit · Not a commercial tool
    </div>
    """,
        unsafe_allow_html=True,
    )
