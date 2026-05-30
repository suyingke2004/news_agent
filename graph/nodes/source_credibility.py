from urllib.parse import urlparse
from graph.state import FactCheckState
from graph.nodes.utils import emit_progress

# Domain credibility tiers
TIER_1_DOMAINS = {
    # International wire services & top-tier outlets
    "reuters.com", "apnews.com", "ap.org",
    "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "nature.com", "science.org", "nejm.org",
    "who.int", "cdc.gov", "nih.gov",
}

TIER_2_DOMAINS = {
    # Major news outlets
    "cnn.com", "cnbc.com", "bloomberg.com", "ft.com",
    "wsj.com", "economist.com", "time.com",
    "aljazeera.com", "npr.org", "pbs.org",
    "abc.net.au", "cbc.ca",
    # Chinese authoritative sources
    "xinhuanet.com", "people.com.cn", "cctv.com",
    "chinadaily.com.cn", "scmp.com",
}

TIER_3_DOMAINS = {
    # Social media and discussion platforms
    "reddit.com", "twitter.com", "x.com",
    "facebook.com", "weibo.com", "zhihu.com",
    "quora.com", "medium.com",
}


def _extract_domain(url: str) -> str:
    """Extract the main domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        # Remove port
        if ":" in domain:
            domain = domain.split(":")[0]
        return domain
    except Exception:
        return ""


def _score_domain(domain: str) -> tuple[float, list[str]]:
    """Score a domain's credibility. Returns (score, reasons)."""
    if not domain:
        return 0.3, ["unknown_domain"]
    
    reasons = []
    
    # Check tier 1
    for t1 in TIER_1_DOMAINS:
        if t1 in domain:
            reasons.append(f"tier_1_source:{t1}")
            return 0.9, reasons
    
    # Check tier 2
    for t2 in TIER_2_DOMAINS:
        if t2 in domain:
            reasons.append(f"tier_2_source:{t2}")
            return 0.75, reasons
    
    # Check tier 3
    for t3 in TIER_3_DOMAINS:
        if t3 in domain:
            reasons.append(f"tier_3_social_media:{t3}")
            return 0.4, reasons
    
    # Unknown domain
    reasons.append("unknown_domain")
    return 0.5, reasons


def source_credibility(state: FactCheckState) -> dict:
    """
    Scores credibility of each source in evidence.
    
    State reads: evidence
    State writes: credibility_scores (via operator.add)
    """
    evidence = state.get("evidence", [])
    
    if not evidence:
        return {"credibility_scores": []}
    
    emit_progress("source_credibility", f"Scoring {len(evidence)} evidence sources...")
    
    # Deduplicate sources by domain
    scored_domains = {}
    
    for ev in evidence:
        domain = _extract_domain(ev.source_url)
        if domain and domain not in scored_domains:
            score, reasons = _score_domain(domain)
            scored_domains[domain] = {
                "source_name": ev.source_name or domain,
                "source_url": ev.source_url,
                "score": score,
                "reasons": reasons,
            }
    
    scores = list(scored_domains.values())
    
    emit_progress(
        "source_credibility",
        f"Scored {len(scores)} unique sources",
    )
    
    return {"credibility_scores": scores}
