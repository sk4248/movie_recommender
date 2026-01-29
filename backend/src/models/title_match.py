from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from difflib import SequenceMatcher

def _normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\(\d{4}\)", "", s)          # remove year like (1995)
    s = re.sub(r"[^a-z0-9\s]", " ", s)       # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass(frozen=True)
class TitleMatch:
    movie_id: int
    title: str
    score: float

def find_best_title_matches(
    movies: pd.DataFrame,
    query: str,
    k: int = 5,
    min_score: float = 0.60
    ) -> List[TitleMatch]:
    """
    Lightweight fuzzy match using SequenceMatcher.
    Returns upto k candidates above min_score
    """
    qn = _normalize_title(query)

    if not qn:
        return []

    # pre compute normalize titles on the fly(MovieLens 100k is small)
    scored: List[Tuple[float,int,str]] = []
    for row in movies[['movie_id','title']].itertuples(index=False):
        tn = _normalize_title(row.title)
        score = SequenceMatcher(None, qn, tn).ratio()
        if score >= min_score:
            scored.append((score, int(row.movie_id), str(row.title)))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [TitleMatch(movie_id=m, title=t, score=s) for (s, m, t) in scored[:k]]