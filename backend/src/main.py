from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {'message': 'Movie Recommender API', 'docs': '/docs'}

class RecommendRequest(BaseModel):
    movies: List[str]
    n: int = 10

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/recommend')
def recommend(req: RecommendRequest):
    # Placeholder for now - we'll replace it with the real recommender logic
    return {
        'seeds': req.movies,
        'recommendations': [
            {
                'title': 'Placeholder title',
                'score': '0.0',
                'explaination': 'API wiring works. recommender logic next'
            }
        ]
    }

