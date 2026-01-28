from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class PopularRecommender:
    min_ratings: int = 50 

def recommend_popular(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    n:int = 10,
    exclude_movie_ids: set[int] | None = None,
    min_ratings: int = 50
) -> pd.DataFrame:
    """
    Popular baseline: rank movies by Bayesian-ish scores:
    - Use average rating but require a minimum count to reduce the noise.
    """
    exclude_movie_ids = exclude_movie_ids or set()

    agg = (
        ratings.groupby('movie_id')['rating']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'rating_count', 'mean': 'rating_mean'})
        .reset_index()
    )


    # Filter + sort
    agg = agg[agg["rating_count"] >= min_ratings]
    agg = agg[~agg["movie_id"].isin(exclude_movie_ids)]
    agg = agg.sort_values(["rating_mean", "rating_count"], ascending=[False, False])

    out = agg.head(n).merge(movies[['movie_id', 'title']], on='movie_id', how='left')
    out['score'] = out['rating_mean'] # keep a consistent API field name
    out['explanation'] = "Popular baseline: high average ratings with enough rating."
    return out[['movie_id', 'title', 'score', 'rating_count', 'explanation']]