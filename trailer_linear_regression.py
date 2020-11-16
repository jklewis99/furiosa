import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv("dbs/trailers_2010s.csv")[['view_count', 'like_count', 'dislike_count', 'tmdb_id']]
    print(dataset.shape)
    dataset = dataset.sort_values(['tmdb_id']).groupby(['tmdb_id']).sum()
    print(dataset.head(10))
    print(type(dataset))

def release_timeline(series_object):
    day_movie_released = series_object['day_released']
    month_movie_released = series_object['month_released']
    year_movie_released = series_object['year_released']
    trailer_release_date = series_object['trailer_release_date']
    if year_movie_released > 0:
        movie_release_date_rcf3339 = datetime.datetime(year_movie_released, month_movie_released, day_movie_released)
        trailer_release_date = datetime.datetime.strptime(trailer_release_date, '%Y-%m-%dT%H:%M:%SZ')
        return movie_release_date_rcf3339 - trailer_release_date
    return np.inf

# Function to scrape only the good trailers
def refine_significant_trailers(trailers):
    # first remove all negative simialrity score values
    trailers = trailers.loc[trailers['similarity_score'] > 0]

if __name__ == "__main__":
    main()