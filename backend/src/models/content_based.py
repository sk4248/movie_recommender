from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict
import numpy as np
import pandas as pd


GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass(frozen=True)
class ContentRec:
    movie_id: int
    title: str
    score: float
    explanation: str


def _build_genre_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      movie_ids: shape (M,)
      X: genre matrix shape (M, G) as float32
    """
    missing = [c for c in GENRE_COLS if c not in movies.columns]
    if missing:
        raise ValueError(f"Movies df missing genre columns: {missing}")

    movie_ids = movies["movie_id"].to_numpy(dtype=np.int32)
    X = movies[GENRE_COLS].to_numpy(dtype=np.float32)

    # Normalize each row for cosine similarity: x / ||x||
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return movie_ids, Xn


def recommend_by_genre_similarity(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    seed_movie_ids: List[int],
    n: int = 10,
    per_seed_k: int = 200,
    exclude_movie_ids: Set[int] | None = None,
) -> List[ContentRec]:
    """
    Content-based recommender using genre cosine similarity.
    Candidate generation: top per_seed_k similar movies per seed (by cosine).
    Ranking: max similarity across seeds + small popularity tie-breaker.
    """
    exclude_movie_ids = set(exclude_movie_ids or set()) | set(seed_movie_ids)

    movie_ids, Xn = _build_genre_matrix(movies)

    # Map movie_id -> row index
    idx_map: Dict[int, int] = {int(mid): i for i, mid in enumerate(movie_ids)}

    seed_idxs = [idx_map[mid] for mid in seed_movie_ids if mid in idx_map]
    if not seed_idxs:
        return []

    # Popularity tie-breaker: rating count per movie (scaled)
    pop = ratings.groupby("movie_id")["rating"].count()
    pop = pop.reindex(movie_ids, fill_value=0).to_numpy(dtype=np.float32)
    pop_scaled = pop / (pop.max() if pop.max() > 0 else 1.0)

    # Candidate generation: for each seed, compute cosine similarity to all
    candidate_scores = {}  # movie_id -> best similarity
    for si in seed_idxs:
        sims = Xn @ Xn[si]  # cosine similarity to seed (since normalized)
        # Get top indices quickly
        top_idx = np.argpartition(-sims, kth=min(per_seed_k, len(sims)-1))[:per_seed_k]
        for j in top_idx:
            mid = int(movie_ids[j])
            if mid in exclude_movie_ids:
                continue
            s = float(sims[j])
            if mid not in candidate_scores or s > candidate_scores[mid]:
                candidate_scores[mid] = s

    if not candidate_scores:
        return []

    # Ranking: primarily by similarity, secondarily by popularity
    items = []
    for mid, sim in candidate_scores.items():
        j = idx_map[mid]
        final_score = sim + 0.05 * float(pop_scaled[j])  # tiny tie-breaker
        items.append((final_score, sim, mid))

    items.sort(reverse=True, key=lambda x: x[0])
    top = items[:n]

    # Build outputs
    movie_title_map = dict(movies[["movie_id", "title"]].itertuples(index=False))
    recs: List[ContentRec] = []
    for final_score, sim, mid in top:
        title = str(movie_title_map.get(mid, "Unknown"))
        recs.append(
            ContentRec(
                movie_id=mid,
                title=title,
                score=float(final_score),
                explanation=f"Similar genres to your picks (genre cosine sim={sim:.2f}).",
            )
        )
    return recs
