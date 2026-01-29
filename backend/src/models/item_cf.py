from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItemItemCF:
    """
    Item-Item Collaborative Filtering (explicit ratings).

    We build a user-item matrix (sparse), normalize ratings by subtracting each user's mean,
    then compute cosine similarity between item vectors.

    After fit():
      - movie_ids: array of movie_ids aligned to item indices
      - idx_map: movie_id -> item index
      - sim: item-item similarity matrix (float32) shape (I, I)
      - pop_scaled: popularity vector aligned to items (for tie-break)
    """
    movie_ids: np.ndarray
    idx_map: Dict[int, int]
    sim: np.ndarray
    pop_scaled: np.ndarray
    title_map: Dict[int, str]

    @staticmethod
    def fit(ratings: pd.DataFrame, movies: pd.DataFrame) -> "ItemItemCF":
        # Map movie_ids to contiguous item indices
        movie_ids = movies["movie_id"].to_numpy(dtype=np.int32)
        idx_map = {int(mid): i for i, mid in enumerate(movie_ids)}

        # Map user_ids to contiguous indices
        user_ids = ratings["user_id"].unique()
        user_ids.sort()
        user_idx = {int(uid): i for i, uid in enumerate(user_ids)}

        # Build sparse ratings matrix R (U x I)
        rows = ratings["user_id"].map(user_idx).to_numpy(dtype=np.int32)
        cols = ratings["movie_id"].map(idx_map).to_numpy(dtype=np.int32)
        vals = ratings["rating"].to_numpy(dtype=np.float32)

        R = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

        # User-mean center: subtract each user's mean from their ratings
        # Compute mean per user from sparse rows
        row_sums = np.asarray(R.sum(axis=1)).ravel()
        row_counts = np.diff(R.indptr)
        row_means = np.divide(row_sums, row_counts, out=np.zeros_like(row_sums), where=row_counts != 0)

        # Create centered matrix Rc with same sparsity (center only observed ratings)
        Rc = R.copy().tocsr()
        # subtract mean from each non-zero entry
        for u in range(Rc.shape[0]):
            start, end = Rc.indptr[u], Rc.indptr[u + 1]
            if start == end:
                continue
            Rc.data[start:end] -= row_means[u]

        # Item-item cosine similarity on centered ratings:
        # similarity between columns (items)
        # Rc is (U x I). We want cosine similarity of item vectors => Rc.T (I x U)
        sim = cosine_similarity(Rc.T, dense_output=True).astype(np.float32)

        # Popularity tie-breaker (rating counts)
        pop = ratings.groupby("movie_id")["rating"].count()
        pop = pop.reindex(movie_ids, fill_value=0).to_numpy(dtype=np.float32)
        pop_scaled = pop / (pop.max() if pop.max() > 0 else 1.0)

        title_map = dict(movies[["movie_id", "title"]].itertuples(index=False))

        return ItemItemCF(
            movie_ids=movie_ids,
            idx_map=idx_map,
            sim=sim,
            pop_scaled=pop_scaled,
            title_map=title_map,
        )

    def recommend(
        self,
        seed_movie_ids: List[int],
        n: int = 10,
        per_seed_k: int = 200,
        exclude_movie_ids: Set[int] | None = None,
    ) -> List[dict]:
        exclude_movie_ids = set(exclude_movie_ids or set()) | set(seed_movie_ids)

        seed_idxs = [self.idx_map[mid] for mid in seed_movie_ids if mid in self.idx_map]
        if not seed_idxs:
            return []

        # Candidate generation + scoring:
        # For each seed, take top per_seed_k similar items.
        # Aggregate score as max(similarity) across seeds (easy + robust),
        # plus tiny popularity tie-break.
        best_sim: Dict[int, float] = {}

        for si in seed_idxs:
            sims = self.sim[si]  # shape (I,)
            # get top indices quickly (includes the seed itself)
            k = min(per_seed_k + 1, sims.shape[0] - 1)
            top_idx = np.argpartition(-sims, kth=k)[:k]
            for j in top_idx:
                mid = int(self.movie_ids[j])
                if mid in exclude_movie_ids:
                    continue
                s = float(sims[j])
                if mid not in best_sim or s > best_sim[mid]:
                    best_sim[mid] = s

        if not best_sim:
            return []

        scored: List[Tuple[float, float, int]] = []
        for mid, s in best_sim.items():
            j = self.idx_map[mid]
            final = s + 0.05 * float(self.pop_scaled[j])
            scored.append((final, s, mid))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:n]

        out = []
        for final, s, mid in top:
            out.append(
                {
                    "movie_id": mid,
                    "title": str(self.title_map.get(mid, "Unknown")),
                    "score": float(final),
                    "explanation": f"Collaborative filtering: users who rated your picks similarly also rated this (item sim={s:.2f}).",
                }
            )
        return out
