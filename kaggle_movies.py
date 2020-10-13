import pandas as pd 
import numpy as np 
dataframe1=pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_credits.csv')
dataframe2=pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_movies.csv')

# join the movies on their ID
dataframe1.columns = ['id','tittle','cast','crew']
dataframe2= dataframe2.merge(dataframe1, on='id')

print(dataframe2.shape) # contains 4803 movies with 23 columns
print(dataframe2['title'].head(5)) 