'''
helper methods
'''

import pandas as pd

def clean_2010s_dataframe(path):
    '''
    method that is used by multiple modules to verify each module is making
    calls with the same data
    '''
    movies_2010s = pd.read_csv(path)
    # get rid of movies with null values:
    movies_2010s.dropna(inplace=True)
    # get revenues that are greater than 0:
    movies_2010s = movies_2010s.loc[movies_2010s['revenue'] > 0]
    num_votes_required = movies_2010s['vote_count'].quantile(0.8)
    watched_movies_2010s = movies_2010s.copy().loc[
        movies_2010s['vote_count'] >= num_votes_required]
    return watched_movies_2010s
