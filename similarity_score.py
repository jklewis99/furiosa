'''
module to give a score to each trailer retrieved by the YouTube API
mainly to eliminate clips and inauthenticate posted videos
'''
import textdistance

def strip_characters(title):
    '''
    method to remove all non-essential characters.
    NOTE: "/" and "\\" are replaced with spaces

    Parameters
    ==========
    title:
        string which needs to be stripped

    Returns
    ==========
    a string with all non-essential characters removed
    '''
    replaced = ""
    for char in title.lower():
        if char in (" ", "/", "\\"):
            replaced += " "
        else:
            val = ord(char)
            if 60 < val < 123 or 47 < val < 58:
                replaced += char
    return replaced

def description_keywords(description_list, keywords):
    '''
    add scores based on keywords in the description

    Parameters
    ==========
    description_list:
        list of words in the description of the video
    keywords:
        list of keywords, such as director and cast members

    Return
    ==========
    integer of count of tose keywords in the description
    '''
    count_contained = 0
    if len(description_list) == 0 or keywords is None:
        return 0
    for word in keywords:
        if word.lower() in description_list:
            count_contained += 1

    return count_contained / len(keywords)

def is_clip(trailer_title_tokens, movie_title_tokens, 
    flags=["clip", "clips", "scene", "scenes", "blooper", "bloopers"]):
    '''
    checks if any terms in flags are in the trailer title list.
    if these flags are contained in the movie_title_tokens, then
    the method defaults to returning 1

    Parameters
    ==========
    trailer_title_tokens:
        list of lowercase tokens in trailer title
    movie_title_tokens:
        list of lowercase tokens in movie title

    Return
    ==========
    -1 if flags are in trailer title and not in movie title, 1 otherwise
    '''

    # TODO: if this {flag}-in-movie-title is a severe issue, it would be smarter to check the
    #       specific flag in the trailer and ensure it is surrounded by the same words as
    #       the words that surround it in the title
    if any(item in movie_title_tokens for item in flags):
        return 1

    contains_flag = any(item in trailer_title_tokens for item in flags)
    if contains_flag:
        return -1
    return 1

def get_similarity_score(title, videos, movie_description, keywords=None):
    '''
    main method that take the list of video_data from a YouTube query
    and sort the queries based on a set of similarities as defined
    by accord index and description keywords

    Parameters
    ==========
    title:
        the title of the movie
    videos:
        list of youtube video objects
    keywords:
        list of top-5 cast, top-5 characters, and director(s)
    movie_description:
        string of description from tmdb

    Return
    ==========
    list of YouTubeVideo objects with similarity score
    '''

    youtube_videos = []
    for video in videos:
        # first, all items will be converted to lowercase and tokenized
        lowercase_title = title.lower()
        movie_desc = strip_characters(movie_description.lower()).split()
        trailer_desc = strip_characters(video.description.lower()).split()
        movie_title = (strip_characters(lowercase_title) + " official trailer").split()
        trailer_title = strip_characters(video.title.lower()).split()

        similarity_score = is_clip(trailer_title, lowercase_title.split()) * 0.5 * (
            textdistance.sorensen_dice(movie_desc, trailer_desc) + textdistance.jaccard(
                movie_title, trailer_title))
        video.set_similarity_score(similarity_score)
        youtube_videos.append(video)

    return youtube_videos
