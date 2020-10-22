# furiosa
An end-to-end Data Science application for predicting revenue for films from the 2010s.

## Data
The data for this project was generated using a set of APIs and databases. All databases can be found [here](/dbs)
### Movies
The initial dataset came from the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/). Only movies that were released in the 2010's decade (2010-2019) were kept. For each of these movies, a request was made to the [TMDB API](https://developers.themoviedb.org/3) for updated or new features for `budget`, `title`, `vote_count`, `vote_average`, `revenue`, `runtime`, `popularity`, and `overview`. After/During these requests, additional requests were made to get information on `credits` and `crew` and `release_dates`. 

### Trailers
To get data for trailers, the [YouTube Data API](https://developers.google.com/youtube/v3) was used. The YouTube API's **search list** method and **Videos list** methods were used to get data on trailers, specifically `title`, `channel_title`, `channel_id`, `description`, `release_date`, `tags`, `view_count`,`like_count`, `dislike_count`, and `comment_count` (features renamed for Python syntax), with a `similarity score` added based on the [similarity_score](/similarity_score) metric.

##

