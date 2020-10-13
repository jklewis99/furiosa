import requests
import numpy as np
from api_info.api_variables import TMDB_API_KEY

URL = "https://api.themoviedb.org/3/movie/"

def get_by_movie_id(movie_id):

    request = f"{URL}{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(request)
    response = response.json()

    budget = response['budget'] if 'budget' in response else np.NaN
    title = response['title'] if 'title' in response else ""
    vote_count = response['vote_count'] if 'vote_count' in response else np.NaN
    vote_average = response['vote_average'] if 'vote_average' in response else np.NaN
    revenue = response['revenue'] if 'revenue' in response else np.NaN
    runtime = response['runtime'] if 'runtime' in response else np.NaN
    popularity = response['popularity'] if 'popularity' in response else np.NaN
    overview = response['overview'] if 'overview' in response else ""

    return {
        'budget': budget,
        'title': title,
        'vote_count': vote_count,
        'vote_average': vote_average,
        'revenue': revenue,
        'runtime': runtime,
        'popularity': popularity,
        "overview": overview
    }

