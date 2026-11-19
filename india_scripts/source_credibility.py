"""
Indian News Source Credibility & Viral Forward Detection Engine
==============================================================
Curated credibility scores for Indian news sources, viral messaging channels,
and known misinformation vectors.

Features:
1. Curated Source Table covering Indian Press (The Hindu, NDTV, PTI, etc.) and
   viral channels (WhatsApp forwards, Telegram, Postcard News, etc.)
2. In-text viral forward detection (e.g. 'Forwarded as received', 'share with all groups', 'कृपया आगे भेजें')
3. Indian viral hoax archetype detector (UNESCO anthem, 500 note midnight ban, 3 green ticks, miracle cures, free recharges)
4. Calibrated Multi-Evidence Probability Fusion
"""

import re
from typing import Tuple, List, Dict, Any

# Adjustment weight for backward compatibility
SOURCE_WEIGHT = 0.45

# ─── 1. Comprehensive Source Credibility Table ──────────────────────────────
# Keys are lowercase domain/channel fragments.
# Score range: 0.0 (fully trustworthy) → 1.0 (known disinfo / unverified viral rumor)
SOURCE_TABLE = {
    # ── Tier 1: Highly Credible Indian & Global Press (0.05 – 0.15) ──────────
    "thehindu":             0.05,
    "thehindu.com":         0.05,
    "hindu":                0.06,
    "indianexpress":        0.07,
    "indianexpress.com":    0.07,
    "ndtv":                 0.08,
    "ndtv.com":             0.08,
    "bbc":                  0.08,
    "bbc.com":              0.08,
    "bbc hindi":            0.08,
    "reuters":              0.06,
    "reuters.com":          0.06,
    "apnews":               0.06,
    "apnews.com":           0.06,
    "pti":                  0.08,   # Press Trust of India
    "ani":                  0.10,   # Asian News International
    "thewire":              0.10,
    "thewire.in":           0.10,
    "scroll":               0.10,
    "scroll.in":            0.10,
    "theprint":             0.12,
    "theprint.in":          0.12,
    "livemint":             0.10,
    "livemint.com":         0.10,
    "mint":                 0.10,
    "businessstandard":     0.10,
    "business-standard":    0.10,
    "financialexpress":     0.12,
    "hindustantimes":       0.12,
    "hindustantimes.com":   0.12,
    "ht":                   0.12,

    # ── Tier 2: Generally Reliable Media (0.18 – 0.35) ──────────────────────
    "economictimes":        0.18,
    "economictimes.indiatimes.com": 0.18,
    "thequint":             0.18,
    "quint":                0.18,
    "cnbctv18":             0.20,
    "tribuneindia":         0.20,
    "timesofindia":         0.22,
    "timesofindia.indiatimes.com": 0.22,
    "toi":                  0.22,
    "indiatoday":           0.20,
    "indiatoday.in":        0.20,
    "outlookindia":         0.22,
    "deccanherald":         0.18,
    "aajtak":               0.26,
    "aajtak.in":            0.26,
    "firstpost":            0.28,
    "abplive":              0.28,
    "abpnews":              0.28,
    "tv9":                  0.30,
    "news18":               0.30,
    "dnaindia":             0.32,
    "zee":                  0.48,

    # ── Tier 3: High-Bias / Sensational Media (0.45 – 0.65) ─────────────────
    "dainikbhaskar":        0.40,
    "amarujala":            0.42,
    "jagran":               0.45,
    "navbharattimes":       0.45,
    "navbharat":            0.48,
    "zeenews":              0.55,
    "republic":             0.65,
    "republicworld":        0.65,
    "republictv":           0.65,
    "sudarshannews":        0.72,
    "newsnation":           0.55,

    # ── Tier 4: Known Disinformation / Clickbait / Satire (0.75 – 0.95) ─────
    "postcard":             0.92,
    "postcard.news":        0.92,
    "opindia":              0.78,
    "rightlog":             0.82,
    "kreately":             0.85,
    "fauxy":                0.88,
    "theonion":             0.88,
    "nationalreport":       0.90,
    "worldnewsdailyreport": 0.92,

    # ── WhatsApp / Social Viral / Messaging Channels (0.85 – 0.95) ──────────
    "whatsapp":             0.92,
    "whatsapp forward":     0.94,
    "whatsapp viral":       0.94,
    "whatsapp group":       0.90,
    "wa forward":           0.92,
    "wa group":             0.90,
    "wa status":            0.90,
    "telegram":             0.88,
    "telegram forward":     0.90,
    "telegram channel":     0.88,
    "facebook forward":     0.88,
    "fb forward":           0.88,
    "facebook":             0.75,
    "fb":                   0.75,
    "facebook post":        0.78,
    "twitter":              0.65,
    "x.com":                0.65,
    "tweet":                0.65,
    "instagram":            0.75,
    "insta":                0.75,
    "reels":                0.78,
    "tiktok":               0.80,
    "sharechat":            0.88,
    "moj":                  0.85,
    "viral video":          0.82,
    "unverified":           0.90,
    "unverified forward":   0.94,
    "chain message":        0.94,
    "family group":         0.92,
    "sms forward":          0.90,
    "social media":         0.78,
    "forward":              0.88,
    "viral":                0.85,
}

DEFAULT_SCORE = 0.50

# ─── 2. Known Indian Viral Hoax Archetypes ──────────────────────────────────
VIRAL_HOAX_PATTERNS = [
    (r"\bunesco\b.*?\b(?:declare|declares|declared|declared as|named|selected)\b.*?\b(?:national anthem|anthem|jana gana mana|best)\b",
     "UNESCO National Anthem Hoax"),
    (r"\b(?:jana gana mana|national anthem)\b.*?\b(?:unesco|best in the world)\b",
     "UNESCO National Anthem Hoax"),
    (r"\bunesco\b.*?\b(?:best prime minister|best pm|best currency|modi best)\b",
     "UNESCO World Best Claim"),
    (r"यूनेस्को.*?(?:घोषित|राष्ट्रगान|सर्वश्रेष्ठ)",
     "UNESCO National Anthem Hoax (Devanagari)"),
    (r"\b(?:three|3)\s+(?:green|red)\s+ticks\b",
     "WhatsApp 3 Green/Red Ticks Surveillance Hoax"),
    (r"\bwhatsapp\b.*?\b(?:calls?|video calls?)\b.*?\b(?:record|monitored|surveillance|recorded|green ticks)\b",
     "WhatsApp Call Recording Rumor"),
    (r"तीन\s+हरे\s+टिक|कॉल\s+रिकॉर्ड",
     "WhatsApp Surveillance Call Hoax (Devanagari)"),
    (r"\b(?:500|2000)\s*(?:rupees?|rs\.?)?\s*(?:note|notes?|currency)\b.*?\b(?:ban|banned|demonetize|demonetized|illegal|invalid|scrapped)\b.*?\b(?:midnight|tonight|tomorrow)\b",
     "Midnight Currency Demonetization Hoax"),
    (r"\b(?:note|notes?)\b.*?\b(?:gps\s*nano\s*chip|nano\s*chip|gps\s*chip)\b",
     "Currency GPS Nanochip Hoax"),
    (r"(?:500|2000).*?(?:नोट\s+बंद|अमान्य|मध्यरात्रि|रात\s*12)",
     "Midnight Note Ban Hoax (Devanagari)"),
    (r"\b(?:boiled\s+ginger|ginger\s+water|garlic\s+water|lemon\s+water|turmeric)\b.*?\b(?:cures?|kills?|cure|100\s*%|100\s*percent)\b.*?\b(?:pneumonia|covid|corona|virus|cancer)\b",
     "Miracle Cure Health Hoax"),
    (r"(?:अदरक|हल्दी|नींबू|लहसुन).*?(?:100%|इलाज|खत्म|कोरोना|कैंसर)",
     "Miracle Cure Health Hoax (Devanagari)"),
    (r"\b(?:free|मुफ्त)\s*(?:recharge|500\s*recharge|laptop|smartphone|5000)\b.*?\b(?:link|click|yojana|yojna|claim)\b",
     "Free Recharge / Phishing Scam"),
    (r"\bnasa\b.*?\b(?:satellite|photo|image|imagery)\b.*?\b(?:diwali|lit up|completely lit)\b",
     "NASA Diwali Satellite Image Hoax"),
    (r"नासा.*?(?:दीवाली|तस्वीर|रोशनी)",
     "NASA Diwali Satellite Hoax (Devanagari)"),
]

# ─── 3. In-Text Forward Markers ─────────────────────────────────────────────
FORWARD_MARKERS = [
    r"\bforward(?:ed)?\s+(?:as\s+received|message|notice|alert)\b",
    r"\bforward\s+this\b",
    r"\bshare\s+(?:with\s+all|maximum|to\s+all|to\s+every|with\s+every)\b",
    r"\bsend\s+to\s+(?:\d+|all|ten|10)\s+(?:people|friends|groups|contacts)\b",
    r"\bdon'?t\s+ignore\b",
    r"\bshare\s+before\s+(?:it\s+is\s+)?deleted\b",
    r"\bviral\s+(?:whatsapp|forward|message|notice)\b",
    r"\bwhatsapp\s+(?:forward|notice|alert|message|rumor)\b",
    r"कृपया\s+(?:आगे\s+भेजें|शेयर\s+करें|सभी\s+ग्रुप)",
    r"सभी\s+(?:ग्रुप|दोस्तों)\s+को\s+भेजें",
    r"फॉरवर्डेड",
    r"वायरल\s+मैसेज",
    r"जरूरी\s+सूचना",
    r"सावधान\s*!",
    r"व्हाट्सएप\s+(?:पर\s+यह\s+मैसेज|फॉरवर्ड|नोटिस)",
]


def get_source_score(source: str) -> Tuple[float, str]:
    """
    Look up credibility score for a given source string or URL.
    Returns (score, human_readable_tier_label).
    """
    if not source or not source.strip():
        return DEFAULT_SCORE, "Unknown (neutral)"

    normalized = source.strip().lower()
    normalized = re.sub(r"^https?://", "", normalized)
    normalized = re.sub(r"^www\.", "", normalized)
    normalized = normalized.split("/")[0]  # Take domain only if URL

    # Match against the table (longest key match first)
    sorted_keys = sorted(SOURCE_TABLE.keys(), key=lambda k: len(k), reverse=True)
    for key in sorted_keys:
        if key in normalized:
            score = SOURCE_TABLE[key]
            return score, _score_to_label(score, key)

    return DEFAULT_SCORE, "Unknown (neutral)"


def _score_to_label(score: float, key: str) -> str:
    """Return a human-readable credibility label."""
    if score <= 0.15:
        return f"{key} — Highly Credible ✅"
    elif score <= 0.35:
        return f"{key} — Generally Reliable 🟡"
    elif score <= 0.65:
        return f"{key} — Questionable Bias ⚠️"
    elif score <= 0.84:
        return f"{key} — High Misinformation Risk 🔴"
    else:
        return f"{key} — Unverified Viral Forward / Hoax Vector 🚨"


def detect_forward_and_hoaxes(text: str, source: str = "") -> Dict[str, Any]:
    """
    Analyzes text and source for forward markers, classic Indian viral hoaxes,
    and sensational urgency phrasing.
    """
    text_lower = text.lower()
    source_lower = source.lower() if source else ""

    matched_hoaxes = []
    for pattern, name in VIRAL_HOAX_PATTERNS:
        if re.search(pattern, text_lower):
            matched_hoaxes.append(name)

    matched_forwards = []
    for pattern in FORWARD_MARKERS:
        m = re.search(pattern, text_lower)
        if m:
            matched_forwards.append(f"In-text: '{m.group(0)}'")

    is_viral_source = any(k in source_lower for k in [
        "whatsapp", "forward", "viral", "telegram", "chain", "group", "social media", "unverified"
    ])
    if is_viral_source:
        matched_forwards.append(f"Source channel: '{source}'")

    has_caps = len(re.findall(r"\b[A-Z]{3,}\b", text)) >= 3
    has_exclamations = len(re.findall(r"[!?]{2,}", text)) >= 1
    has_urgency = bool(re.search(r"\b(?:urgent|alert|emergency|breaking|warning|आपातकालीन|चेतावनी)\b", text_lower))

    return {
        "matched_hoaxes": list(dict.fromkeys(matched_hoaxes)),
        "matched_forwards": list(dict.fromkeys(matched_forwards)),
        "is_forward": len(matched_forwards) > 0,
        "is_hoax": len(matched_hoaxes) > 0,
        "has_urgency": has_caps or has_exclamations or has_urgency,
    }


def compute_calibrated_fake_probability(
    raw_fake_prob: float,
    source: str,
    text: str,
    sentiment_neg_prob: float = 0.0
) -> Dict[str, Any]:
    """
    Multi-evidence fusion combining:
    1. Neural model raw output (raw_fake_prob)
    2. Curated source credibility (source_score)
    3. Indian viral hoax archetype detection
    4. In-text WhatsApp forward markers
    5. Emotional panic / sensational urgency signals
    """
    source_score, source_label = get_source_score(source)
    analysis = detect_forward_and_hoaxes(text, source)

    is_hoax = analysis["is_hoax"]
    is_forward = analysis["is_forward"]
    matched_hoaxes = analysis["matched_hoaxes"]
    has_urgency = analysis["has_urgency"]

    # If text is clearly an unverified forward but source was left blank, calibrate source
    if is_forward and (not source or source_score == DEFAULT_SCORE):
        source_score = 0.90
        source_label = "In-Text Viral Forward Detected 📱 (Score 0.90)"

    # ── Rule 1: High Credibility Tier 1 Anchor ───────────────────────────────
    # If published by The Hindu, BBC, NDTV, etc. and not a known viral hoax
    if source_score <= 0.15 and not is_hoax:
        fake_p = min(0.18, raw_fake_prob * 0.20 + source_score * 0.80)
        risk_level = "Verified Journalism"

    # ── Rule 2: Classic Viral Hoax Match ─────────────────────────────────────
    # If recognized viral hoax matched (e.g. UNESCO anthem, 500 note ban midnight, 3 green ticks)
    elif is_hoax:
        base = 0.92
        if is_forward or source_score >= 0.75:
            base = 0.96
        fake_p = max(base, min(0.99, base + 0.03 * has_urgency))
        risk_level = f"Confirmed Viral Hoax ({matched_hoaxes[0]})"

    # ── Rule 3: WhatsApp / Unverified Forward Channel ────────────────────────
    elif is_forward or source_score >= 0.85:
        urgency_boost = 0.08 if has_urgency else 0.0
        neg_boost = 0.05 if sentiment_neg_prob > 0.50 else 0.0
        blended = (source_score * 0.70) + (raw_fake_prob * 0.30) + urgency_boost + neg_boost
        fake_p = max(0.72, min(0.98, blended))
        risk_level = "High-Risk Unverified Viral Forward"

    # ── Rule 4: Known Disinformation / Clickbait Media ───────────────────────
    elif source_score >= 0.75:
        fake_p = max(0.80, (source_score * 0.65) + (raw_fake_prob * 0.35))
        risk_level = "Disinformation / Partisan Portal"

    # ── Rule 5: Generally Reliable / Standard Source ─────────────────────────
    elif source_score <= 0.35:
        fake_p = (raw_fake_prob * 0.40) + (source_score * 0.60)
        risk_level = "Generally Reliable Media"

    # ── Rule 6: Neutral / Unknown Source (Default) ───────────────────────────
    else:
        shift = 0.0
        if has_urgency:
            shift += 0.10
        if sentiment_neg_prob > 0.60:
            shift += 0.08
        fake_p = max(0.02, min(0.98, raw_fake_prob + shift))
        risk_level = "Standard Text Analysis"

    fake_p = float(max(0.01, min(0.99, fake_p)))
    verdict = "FAKE" if fake_p >= 0.50 else "REAL"
    confidence = fake_p if verdict == "FAKE" else (1.0 - fake_p)

    return {
        "verdict": verdict,
        "confidence": f"{confidence * 100:.1f}%",
        "fake_probability": f"{fake_p * 100:.1f}%",
        "real_probability": f"{(1.0 - fake_p) * 100:.1f}%",
        "adjusted_fake_prob": fake_p,
        "source_score": round(source_score, 2),
        "source_credibility": source_label,
        "risk_level": risk_level,
        "matched_hoaxes": matched_hoaxes,
        "is_forward": is_forward,
        "matched_forwards": analysis["matched_forwards"],
    }


def apply_source_adjustment(raw_fake_prob: float, source_score: float, text: str = "", sentiment_neg: float = 0.0) -> float:
    """
    Maintained for backward compatibility. Uses calibrated fusion if text is provided,
    otherwise applies calibrated weighted shift.
    """
    if text:
        res = compute_calibrated_fake_probability(raw_fake_prob, "", text, sentiment_neg)
        return res["adjusted_fake_prob"]

    # If only raw probability and score are passed without text
    if source_score <= 0.15:
        return float(min(0.20, raw_fake_prob * 0.3 + source_score * 0.7))
    elif source_score >= 0.80:
        return float(max(0.70, raw_fake_prob * 0.3 + source_score * 0.7))
    else:
        adjustment = SOURCE_WEIGHT * (source_score - 0.5)
        return float(max(0.0, min(1.0, raw_fake_prob + adjustment)))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    test_cases = [
        ("UNESCO declares Jana Gana Mana as best national anthem in emergency summit meeting.", "whatsapp forward", 0.004),
        ("Government to monitor all WhatsApp video calls with three green ticks starting tomorrow.", "whatsapp forward", 0.026),
        ("Forward this WhatsApp notice: Drinking boiled ginger water cures 100 percent of viral pneumonia.", "", 0.828),
        ("The Hindu reports ISRO launched oceanographic satellite from Sriharikota.", "thehindu.com", 0.956),
    ]
    for text, src, raw in test_cases:
        res = compute_calibrated_fake_probability(raw, src, text)
        print(f"[{res['verdict']}] {res['fake_probability']} (Confidence: {res['confidence']}) | {res['risk_level']}")
