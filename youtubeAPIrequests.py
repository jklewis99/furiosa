import googleapiclient.discovery
import googleapiclient.errors
from similarity_score import get_similarity_score
from youtube_video import YouTubeVideo

def get_youtube_video_statistics(tmdb_id, youtube_ids):
    '''
    helper method to search youtube for a specific video by a video's
    id. This query will cost 1 unit.

    Parameters
    ==========
    youtube_ids:
        list of youtube ids or one youtube id

    Return
    ==========
    a list of YouTubeVideo objects
    '''
    
    api_service_name = "youtube"
    api_version = "v3"
    
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version) #, credentials=credentials)

    request = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(youtube_ids)
    )
    response_items = request.execute()['items']
    # print(type(response))
    return [YouTubeVideo(response['id'], refactor_escape_characters(response['snippet']['title']),
        response['snippet']['channelTitle'], response['snippet']['channelId'],
        response['snippet']['description'], response['snippet']['publishedAt'],
        response['snippet']['tags'] if 'tags' in response['snippet'] else [],
        response['statistics']['viewCount'] if 'viewCount' in response['statistics'] else 0, 
        response['statistics']['likeCount'] if 'likeCount' in response['statistics'] else 0,
        response['statistics']['dislikeCount'] if 'dislikeCount' in response['statistics'] else 0, 
        response['statistics']['commentCount'] if 'commentCount' in response['statistics'] else 0,
        tmdb_id)
        for response in response_items]

def search_youtube(tmdb_id, title, release_date=None):
    '''
    method to search youtube with the query: "title + ' trailer'".
    This query will cost 100 units so searches will need to be
    split across a few days

    Parameters
    ==========
    title:
        movie title
    release_date:
        date that the movie was released, formatted RFC 3339 date-time
        value (1970-01-01T00:00:00Z)

    Return
    ==========
    a list of YouTubeVideo objects
    '''
    api_service_name = "youtube"
    api_version = "v3"
    
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version) #, credentials=credentials)

    if release_date is None:
        release_date = "2020-10-22T00:00:00Z"

    request = youtube.search().list(
        part="snippet",
        maxResults=6,
        type="video",           # exclude channels and playlists
        videoDuration="short",  # attempt to not get film analysis
        q=title + " trailer",   # search for trailers of this movie
        publishedBefore=release_date,
        publishedAfter="2009-01-01T00:00:00Z"
    )
    response = request.execute()
    youtube_ids = [video['id']['videoId'] for video in response['items']]
    return get_youtube_video_statistics(tmdb_id, youtube_ids)

def refactor_escape_characters(string):
    '''
    method to refactor youtube escape characters. YouTube uses
    special characters to format its strings.

    Parameters
    ==========
    string:
        the sequence of characters that need to be replaced

    Return
    ==========
    the updated string
    '''

    youtube_special_characters = {
        "&quot;": "\"",
        "&amp;": "&",
        "&#39;": "'"
    }

    for key in youtube_special_characters:
        if key in string:
            string = string.replace(key, youtube_special_characters[key])

    return string

def compare_api_responses():
    '''
    though tmdb does have a video with which to associate the movie
    the video associated is not always the most popular.
    '''
    import pandas as pd
    from tmdbAPIrequests import appended_movie_info
    movie_ids = pd.read_csv("movies-from-2010s.csv")[['tmdb_id', 'title']].head(10)
    ids_from_tmdb = []
    ids_from_youtube = []
    for _, id in movie_ids['tmdb_id'].iteritems():
        # ids_from_youtube.append(search_youtube(id))
        ids_from_tmdb.append(appended_movie_info(id))
    for _, name in movie_ids['title'].iteritems():
        ids_from_youtube.append(search_youtube(name))
        # ids_from_tmdb.append(appended_movie_info(id))

    tmdb = "TMDB"
    yt = "YouTube"
    count_correct = 0
    print(f"{tmdb:20} | {yt:20}" )
    for t, y in zip(ids_from_tmdb, ids_from_youtube):
        ugh = y['youtube_id'] if y['youtube_id'] is not None else "No Id"
        
        try:
            print(f"{t:20} | {ugh:20}" )
        except:
            print("failed")
        if t == y:
            count_correct += 1
    tmdb_stats = []
    for response in ids_from_tmdb:
        try:
            tmdb_stats.append(get_youtube_video_statistics(response))
        except:
            tmdb_stats.append(None)
            print("ERROR at key:", response)

    print("====================================================")
    print(f"\n{tmdb:20} | {yt:20}" )
    for t, y in zip(tmdb_stats, ids_from_youtube):
        try:
            print(f"{t['viewCount']:20} | {y['viewCount']:20}")
        except:
            print("failed")
    print(count_correct)

if __name__ == "__main__":
    compare_api_responses()