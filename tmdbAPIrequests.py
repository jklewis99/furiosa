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

def appended_movie_info(movie_id):
    # appendn_to_response creates a new key for each appended response
    request = f"{URL}{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos,reviews"
    response = requests.get(request)
    response = response.json()

    # for json_object in response:
    #     if json_object == 'credits':
    #         print("\n=========Credits=========\n")
    #         print(response['credits'])
    #     elif json_object  == 'videos':
    #         print("\n=========Videos=========\n")
    #         print(response['videos'])
    #     elif json_object  == 'reviews':
    #         print("\n=========Reviews=========\n")
    #         print(response['reviews'])

    return response['videos']['results'][0]['key'] if len(response['videos']['results']) > 0 else None
    # return {
    #     "credits": response['credits'],

    # }

def main():
    # test for the movie SHutter Island: 74458
    appended_movie_info(23023)


main()

