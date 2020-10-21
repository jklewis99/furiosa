import heapq

def accord_index(actual, expected):
    '''
    calculate the accord score of two strings of trailer
    titles. These strings will become sets stripped of special
    characters
    '''
    actual = strip_characters(actual).split()
    expected = strip_characters(expected).split()
    intersection = 0
    for word in actual:
        if word in expected:
            intersection += 1

    union = len(actual) + len(expected) - intersection

    return 1.0 * intersection / union

def strip_characters(title):
    replaced = ""
    for char in title.lower():
        if char == " ":
            replaced += char
        else:
            val = ord(char)
            if val > 60 and val < 123 or val > 47 and val < 58:
                replaced += char
    return replaced

def description_keywords(description, keywords):
    '''
    add scores based on keywords in the description

    Parameters
    ==========
    description:
        string of the description of the video
    keywords:
        list of keywords, such as director and cast members

    Return
    ==========
    integer of count of tose keywords in the description
    '''
    count_contained = 0
    description_list = description.split()
    if description_list == 0:
        return 0
    for word in description_list:
        if word in keywords:
            count_contained += 1
    
    return 1.0 * count_contained / len(description_list)

def is_clip(title):
    title_list = title.lower().split()
    if "clip" in title_list:
        return -1
    return 1

def get_similarity_score(title, videos, keywords):
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

    Return
    ==========
    priority_queue of youtube videos based on their similarity score
    '''

    priority_queue = []
    for video in videos:
        similarity_score = is_clip(video.title) * (description_keywords(video.description, keywords)
            + accord_index(title + " official trailer", video.title))
        video.set_similarity_score(similarity_score)
        heapq.heappush(priority_queue, video)

    return priority_queue

if __name__ == "__main__":
    print(accord_index(["incredible", "hulk", "official", "trailer"],
                 ["the", "incredible", "hulk", "official", "trailer"]))