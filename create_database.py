import pandas as pd 
import datetime
from tmdbAPIrequests import get_by_movie_id
COUNT = 0


def get_year(series_object):
    # title is currently formatted as "name (XXXX)"
    former_title = series_object['title']
    year = former_title[-6:] if len(former_title) > 5 else former_title
    if "(" in year:
        year = int(year[year.index("(")+1:year.index(")")])
    else:
        year = 0

    return year

def update_dataframe(series_object, total):
    global COUNT
    tmdb_id = series_object['tmdb_id']
    response = get_by_movie_id(tmdb_id)

    series_object['budget'] = response['budget']
    series_object['title'] = response['title']
    series_object['vote_count'] = response['vote_count']
    series_object['vote_average'] = response['vote_average']
    series_object['revenue'] = response['revenue']
    series_object['runtime'] = response['runtime']
    series_object['popularity'] = response['popularity']
    series_object['overview'] = response['overview']
    COUNT += 1
    if COUNT % 1000 == 0:
        print(COUNT, "/", total)
    return series_object

def main():
    start = datetime.datetime.now()
    movies = pd.read_csv('data/ml-25m/movies.csv')
    links = pd.read_csv('data/ml-25m/links.csv')
    # genres = pd.read_csv('data/ml-25m/tags.csv')
    print("Time to read:", datetime.datetime.now() - start)
    # join the movies on their ID
    movies = movies.merge(links, on='movieId')
    # we will now have a table of movie_id, title, genres, imdb_id, tmdb_id
    movies.columns = ['movie_id', 'title', 'genres', 'imdb_id', 'tmdb_id']

    movies['year'] = movies.apply(get_year, axis=1)
    movies.dropna(inplace=True) # drop values that do not have a tmdb_id
    movies['tmdb_id'] = movies['tmdb_id'].astype(int)
    movies_2010s = movies.copy().loc[movies['year'] >= 2010]
    total = movies_2010s.shape[0]
    movies_2010s = movies_2010s.apply(lambda x: update_dataframe(x, total), axis=1)
    movies_2010s.to_csv("movies_from_2010s.csv", index=False)
    print("Time to request and build:", datetime.datetime.now() - start)

if __name__ == '__main__':
    main() 