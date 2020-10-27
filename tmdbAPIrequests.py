'''
module containing all functions to make calls to the TMDB API
'''
import requests
import numpy as np
from api_info.api_variables import TMDB_API_KEY

URL = "https://api.themoviedb.org/3/movie/"

def get_by_movie_id(movie_id):
    '''
    request that will return only budget, title, vote_count, vote_average, revenue, runtime,
    popularity, and overview 
    '''
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

def appended_movie_info(movie_id):
    # append_to_response creates a new key for each appended response
    request = f"{URL}{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,release_dates"
    response = requests.get(request)
    response = response.json()

    # get the release date for the United States theatrical release
    release_date_usa = None
    for releases in response['release_dates']['results']:
        if releases['iso_3166_1'] == "US":
            for release in releases['release_dates']:
                if release['type'] == 3:
                    release_date_usa = release['release_date']

    return {
        "credits": response['credits'],
        "release_date": release_date_usa
    }

def append_to_response(movie_id, appended_requests):
    '''
    method to append to normal query to movie

    Parameters
    ==========
    appended_requests:
        list of strings specifying the requests to append


    Returns
    ==========
    object containing default movie details and all requested keys
    '''
    # Acceptable keys, as specified https://developers.themoviedb.org/3/movies/get-movie-details
    acceptable_keys = ["account_states", "alternative_titles", "changes", "credits", \
        "external_ids", "images", "keywords", "release_dates", "videos", "translations", \
        "recommendations", "similar_movies", "reviews", "lists", "latest", "now_playing", \
        "popular", "top_rated", "upcoming"]
    for item in appended_requests:
        if item not in acceptable_keys:
            raise ValueError(f"{item} is not an accpetable request to append to response")

    # append_to_response creates a new key for each appended response
    appended_string = ",".join(appended_requests)
    request = f"{URL}{movie_id}?api_key={TMDB_API_KEY}&append_to_response={appended_string}"
    response = requests.get(request)
    return response.json()

def get_reviews(movie_id):
    # TODO: find different source for reviews, as the TMDB reviews are limited
    pass
