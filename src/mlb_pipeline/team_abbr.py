"""
Canonical team abbreviation normalization for the MLB pipeline.

All parsers and crawlers should import from here rather than maintaining
their own normalization dicts.
"""

# MSF API abbreviation variants → canonical form
MSF_ABBR_NORMALIZE: dict[str, str] = {
    "SFG": "SF",
    "SDP": "SD",
    "KCR": "KC",
    "TBR": "TB",
    "WSN": "WAS",
    "CHW": "CWS",
}

# MLB Stats API abbreviation variants → canonical form
STATSAPI_ABBR_NORMALIZE: dict[str, str] = {
    "AZ":  "ARI",
    "WSH": "WAS",
}

# Combined: checks both MSF and Stats API variants
_ALL_NORMALIZE: dict[str, str] = {**MSF_ABBR_NORMALIZE, **STATSAPI_ABBR_NORMALIZE}


def norm_abbr(abbr: str) -> str:
    """Normalize a team abbreviation from any API source."""
    a = (abbr or "").strip().upper()
    return _ALL_NORMALIZE.get(a, a)


def norm_statsapi(abbr: str) -> str:
    """Normalize Stats API-specific abbreviations."""
    return STATSAPI_ABBR_NORMALIZE.get(abbr, abbr)
