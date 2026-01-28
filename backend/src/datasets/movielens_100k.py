from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# In this file, we load the data from u.data and u.item files
# Adds columns to both these tables and load them together into the MovieLens100k

@dataclass(frozen=True)
class MovieLens100k:
    ratings: pd.DataFrame # user_d, movie_id, rating and timestamp
    movies: pd.DataFrame # movie_id, title, release_date, ... + genres

def load_movielens_100k(data_dir: str | Path) -> MovieLens100k:
    data_dir = Path(data_dir)
    root = data_dir / "ml-100k"
    if not root.exists():
        raise FileNotFoundError(f"Expected folder not found: {root}")

    # Ratings table
    ratings = pd.read_csv(
        root / "u.data",
        sep="\t",
        names = ['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    movie_cols = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    movies = pd.read_csv(
        root / "u.item",
        sep="|",
        names=movie_cols,
        encoding="latin-1",  # MovieLens 100K uses latin-1
        engine="python",
    )

    return MovieLens100k(ratings=ratings, movies=movies)