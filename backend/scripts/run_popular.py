from pathlib import Path
import sys

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.movielens_100k import load_movielens_100k
from src.models.popular import recommend_popular

if __name__ == "__main__":
    data = load_movielens_100k(Path("data/raw"))
    recs = recommend_popular(
        ratings=data.ratings,
        movies=data.movies,
        n=10,
        exclude_movie_ids=set(),
        min_ratings=50,
    )
    print(recs.to_string(index=False))
