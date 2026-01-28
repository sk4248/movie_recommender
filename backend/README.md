# Movie Recommender Backend

FastAPI-based recommendation engine serving movie recommendations via REST API.

## Structure

```
backend/
├── src/
│   ├── datasets/
│   │   └── movielens_100k.py    # MovieLens 100K data loader
│   ├── models/
│   │   └── popular.py            # Popularity-based recommender
│   └── main.py                   # FastAPI application
├── scripts/
│   └── run_popular.py            # Standalone script to test recommender
└── data/raw/
    └── ml-100k/                  # MovieLens 100K dataset
```

## API Endpoints

### Health Check
```
GET /health
```

### Get Recommendations
```
POST /recommend
Body: {
  "movies": ["Movie Title 1", "Movie Title 2"],
  "n": 10
}
```

## Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pandas

# Run server
uvicorn src.main:app --reload
```

Server runs on http://127.0.0.1:8000

## Testing Recommender

Run the standalone script to test the popular recommender:

```bash
cd backend
python scripts/run_popular.py
```

## Current Implementation

**Popular Baseline Recommender**
- Aggregates ratings by movie_id
- Filters movies with minimum 50 ratings
- Ranks by average rating (mean) and rating count
- Returns top N movies with scores and explanations

## Next Steps

- Add collaborative filtering models
- Implement user-based and item-based similarity
- Add genre-based content filtering
- Cache recommendations for performance
