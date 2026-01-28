# Movie Recommender System

A full-stack movie recommendation application built with FastAPI and React, designed to explore and implement various recommendation algorithms using the MovieLens 100K dataset.

## Project Goals

### Phase 1: Baseline Implementation (Current)
- Build a working end-to-end recommendation system
- Implement a popularity-based baseline recommender
- Establish API contracts and data loading pipeline
- Create a simple, functional UI for testing recommendations

### Phase 2: Personalized Recommendations (Planned)
- Implement collaborative filtering algorithms
- Use seed movies provided by users to personalize recommendations
- Add user preference learning and similarity calculations
- Explore content-based filtering using movie genres and metadata

### Phase 3: Advanced Techniques (Future)
- Hybrid recommendation approaches combining multiple algorithms
- Matrix factorization and deep learning models
- Real-time recommendation updates
- A/B testing framework for comparing algorithm performance

## Architecture

```
movie_recommender/
├── backend/          # FastAPI server with recommendation engine
│   ├── src/
│   │   ├── datasets/    # Data loading utilities (MovieLens 100K)
│   │   ├── models/      # Recommendation algorithms
│   │   └── main.py      # FastAPI application entry point
│   └── data/raw/        # MovieLens 100K dataset
└── frontend/         # React + TypeScript UI
    └── src/
        └── App.tsx      # Main application component
```

## Dataset

Using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/):
- 100,000 ratings (1-5 stars)
- 943 users
- 1,682 movies
- Demographic info and movie genres included

## Current Features

- **Popular Baseline Recommender**: Ranks movies by average rating with a minimum rating count threshold (50+ ratings) to balance popularity and quality
- **REST API**: `/recommend` endpoint accepts seed movies and returns top N recommendations
- **React Frontend**: Simple interface for entering movie preferences and viewing recommendations
- **CORS-Enabled**: Backend configured for local development

## Getting Started

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install fastapi uvicorn pandas
uvicorn src.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 to use the application.

## API Usage

**POST /recommend**
```json
{
  "movies": ["Inception", "Interstellar"],
  "n": 10
}
```

Response:
```json
{
  "seeds": ["Inception", "Interstellar"],
  "recommendations": [
    {
      "movie_id": 50,
      "title": "Star Wars (1977)",
      "score": 4.35,
      "explanation": "Popular baseline: high average ratings with enough rating."
    }
  ]
}
```

## Next Steps

1. Implement user-item collaborative filtering
2. Use seed movies to filter recommendations by similar user preferences
3. Add genre-based content filtering
4. Improve UI with movie posters and detailed information
5. Add evaluation metrics (precision, recall, NDCG)

## License

Educational project for exploring recommendation systems.
