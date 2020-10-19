import pandas as pd 
import numpy as np 
dataframe1=pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_credits.csv')
dataframe2=pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_movies.csv')

# join the movies on their ID
print(dataframe1.head(5)) # contains 4803 movies with 23 columns
dataframe1.columns = ['id','title','cast','crew']
print(dataframe1.head(5))
dataframe2= dataframe2.merge(dataframe1, on='id')


print(dataframe2.head(5)) 