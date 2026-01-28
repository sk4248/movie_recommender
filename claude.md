# Claude Development Notes

## Project Context

This is a movie recommendation system built as a learning project to explore different recommendation algorithms. The project uses the MovieLens 100K dataset and is structured as a full-stack application.

## Current State

### Completed
- Basic project structure with FastAPI backend and React frontend
- MovieLens 100K dataset loaded and parsed
- Popularity-based baseline recommender implemented
- REST API with `/recommend` endpoint
- Simple React UI for testing recommendations
- CORS middleware configured

### In Progress
- The system currently ignores seed movies provided by users (see backend/src/main.py:41)
- Recommendations are purely based on global popularity

### Known Issues
- CORS configuration may need adjustment for 127.0.0.1 vs localhost
- No personalization yet - all users get the same recommendations
- No caching of recommendations
- No error handling for missing movies in seed list

## Architecture Decisions

### Data Loading
- Data loaded once on application startup and stored in global `DATA` variable
- Uses pandas DataFrames for efficient aggregation and filtering
- MovieLens 100K format: tab-separated u.data, pipe-separated u.item

### Recommendation Algorithm
- **Popular Baseline** (current): Bayesian-style approach using average ratings with minimum count threshold
  - Minimum 50 ratings required to reduce noise
  - Sorts by mean rating, then by rating count
  - Simple, deterministic, and fast

### API Design
- Request includes seed movies and count (n)
- Response includes seeds echoed back + recommendations with movie_id, title, score, explanation
- Frontend sends movie titles but backend doesn't use them yet

## Next Development Phases

### Phase 2: Personalized Recommendations
1. **Collaborative Filtering**
   - User-based: Find similar users based on rating patterns
   - Item-based: Find similar movies based on co-ratings
   - Use seed movies to generate personalized recommendations

2. **Content-Based Filtering**
   - Use genre information from movies DataFrame
   - Calculate similarity based on genre overlap
   - Combine with seed movies for better recommendations

### Phase 3: Advanced Features
- Hybrid models combining collaborative + content-based
- Matrix factorization (SVD, ALS)
- Deep learning approaches (neural collaborative filtering)
- Evaluation framework with metrics (precision@k, recall@k, NDCG)
- A/B testing infrastructure

## Technical Notes

### Dependencies
- **Backend**: FastAPI, uvicorn, pandas, pydantic
- **Frontend**: React 19, TypeScript, Vite
- **Dataset**: MovieLens 100K (included in repo)

### Development Workflow
- Backend runs on port 8000 (uvicorn with --reload)
- Frontend runs on port 5173 (Vite dev server)
- CORS configured to allow frontend origin

### Code Style
- Type hints used throughout Python code
- Dataclasses for immutable data structures
- Functional approach for recommendation algorithms
- React hooks (useState) for state management

## Files to Watch

### Critical Files
- `backend/src/main.py` - API endpoints and application setup
- `backend/src/models/popular.py` - Current recommendation logic
- `backend/src/datasets/movielens_100k.py` - Data loading
- `frontend/src/App.tsx` - UI and API integration

### Future Model Files
- `backend/src/models/collaborative.py` - Collaborative filtering (planned)
- `backend/src/models/content_based.py` - Content-based filtering (planned)
- `backend/src/models/hybrid.py` - Hybrid approaches (planned)

## Testing Strategy

### Current Testing
- `backend/scripts/run_popular.py` - Standalone script to test recommender
- Manual testing via frontend UI

### Future Testing
- Unit tests for each recommendation algorithm
- Integration tests for API endpoints
- Evaluation scripts with held-out test data
- Performance benchmarks for different algorithms

## Common Tasks

### Add a New Recommender
1. Create new file in `backend/src/models/`
2. Implement function with signature: `(ratings, movies, n, exclude_movie_ids, **kwargs) -> DataFrame`
3. Update `main.py` to use new recommender
4. Test with `scripts/` standalone script

### Update API
1. Modify request/response models in `main.py`
2. Update frontend TypeScript types in `App.tsx`
3. Update API documentation in README

### Work with Data
- Ratings: user_id, movie_id, rating (1-5), timestamp
- Movies: movie_id, title, release_date, genres (one-hot encoded)
- Dataset is small enough to fit in memory (100K ratings, 1682 movies)
