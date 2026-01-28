from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

from src.datasets.movielens_100k import load_movielens_100k
from src.models.popular import recommend_popular

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

        # For baseline, we ignore the seed titles (we’ll use them next phase)
    df = recommend_popular(
        ratings=DATA.ratings,
        movies=DATA.movies,
        n=req.n,
        exclude_movie_ids=set(),
        min_ratings=50,
    )

    return {
        'seeds': req.movies,
        'recommendations': [
           {
             'movie_id': int(row.movie_id),
             'title': row.title,
             'score': float(row.score),
             'explanation': row.explanation
           }
           for row in df.itertuples(index=False)
        ],
   }
