# furiosa
An end-to-end Data Science application for predicting revenue for films from the 2010s.

## Getting Started
Ensure you have Python 3.5 or greater installed. You can use pip or anaconda. You can download the latest version [here](https://www.python.org/downloads/).
#### 1. Clone the repository
Navigate to the folder in which you want to store this repository. Then clone the repository and change directory to the repository:
```
git clone https://github.com/jklewis99/furiosa.git
cd furiosa
```
#### 2. Activate a virtual environment:

##### With pip:
Windows
```
py -m venv [ENV_NAME]
.\[ENV_NAME]\Scripts\activate
```
Linux/Mac
```
python3 -m venv [ENV_NAME]
source [ENV_NAME]/bin/activate
```

##### With conda:
```
conda update conda
conda create -n [ENV_NAME]
```
#### 3. Install the requirements:
```
pip install -r requirements.txt
```
## Data
The data for this project was generated using a set of APIs and databases. All databases can be found [here](/dbs).
### Movies
The initial dataset came from the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/). Only movies that were released in the 2010's decade (2010-2019) were kept. For each of these movies, a request was made to the [TMDB API](https://developers.themoviedb.org/3) for updated or new features for `budget`, `title`, `vote_count`, `vote_average`, `revenue`, `runtime`, `popularity`, and `overview`. After/During these requests, additional requests were made to get information on `credits` and `crew` and `release_dates`. 

### Trailers
To get data for trailers, the [YouTube Data API](https://developers.google.com/youtube/v3) was used. The YouTube API's **search list** method and **Videos list** methods were used to get data on trailers, specifically `title`, `channel_title`, `channel_id`, `description`, `release_date`, `tags`, `view_count`,`like_count`, `dislike_count`, and `comment_count` (features renamed for Python syntax), with a `similarity score` added based on the [similarity_score](/similarity_score.py) metric.

## Setting up YouTube API calls
The main API requests used in this project can be found in the [`youtubeAPIrequests.py`](/youtubeAPIrequests.py) file. In order for this file to work on your computer without access tokens and refresh tokens (and signing in at every execution of the file), there are a few steps to follow:

#### 1. Set up the YouTube Data API on your Google account
Follow directions at the [YouTube Data API Overview](https://developers.google.com/youtube/v3/getting-started) page to get started.

#### 2. Set up the Google Cloud API
Follow directions at the [Getting Started with authentication](https://cloud.google.com/docs/authentication/getting-started) page to get started. When you have the JSON file that identifies the credientials to your application, set up your Environment Variable (System Properties -> Advanced -> Environment Variables -> User variables for {USER} -> New) for `GOOGLE_APPLICATION_CREDENTIALS` to the path where your application's JSON file is saved. If you do not wish to set up this environment variable globally, you can set it up at the beginning of each shell session instead. These instuctions can be found [here](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable).

#### 3. Verify authentication
Run the following commands in the console at the root of the furiosa directory:
```
cd examples
python youtube_api_test.py
```
This should return the following:
```
Toy Story 3: Trailer - Walt Disney Studios
```