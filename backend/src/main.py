from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

from src.datasets.movielens_100k import load_movielens_100k

from src.models.title_match import find_best_title_matches
from src.models.content_based import recommend_by_genre_similarity


app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    movies: List[str]
    n: int = 10


DATA = None

@app.on_event('startup')
def startup():
    global DATA
    DATA = load_movielens_100k(Path('data/raw'))

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    # 1) Resolve seed titles -> movie_ids
    seed_movie_ids: list[int] = []
    resolved = []
    unresolved = []

    for q in req.movies:
        matches = find_best_title_matches(DATA.movies, q, k=3, min_score=0.62)
        if not matches:
            unresolved.append(q)
            continue
        best = matches[0]
        seed_movie_ids.append(best.movie_id)
        resolved.append({"query": q, "matched_title": best.title, "movie_id": best.movie_id, "match_score": best.score})

    # If nothing resolves, fall back to popular baseline behavior (for UX)
    if not seed_movie_ids:
        return {
            "seeds": req.movies,
            "resolved": resolved,
            "unresolved": unresolved,
            "recommendations": [],
            "note": "No seed titles matched the dataset. Try adding year, e.g. 'Toy Story (1995)'.",
        }

    # 2) Recommend via content similarity
    recs = recommend_by_genre_similarity(
        movies=DATA.movies,
        ratings=DATA.ratings,
        seed_movie_ids=seed_movie_ids,
        n=req.n,
        per_seed_k=300,
        exclude_movie_ids=set(seed_movie_ids),
    )

    return {
        "seeds": req.movies,
        "resolved": resolved,
        "unresolved": unresolved,
        "recommendations": [
            {"movie_id": r.movie_id, "title": r.title, "score": r.score, "explanation": r.explanation}
            for r in recs
        ],
    }

