class YouTubeVideo():
    def __init__(self, youtube_id, title, channel_title, channel_id, description,
        tags, view_count, like_count, dislike_count, comment_count, tmdb_id=None,
        similarity_score=None):
        self.youtube_id = youtube_id
        self.title = title
        self.channel_title = channel_title
        self.channel_id = channel_id
        self.description = description
        self.tags = tags
        self.view_count = view_count
        self.like_count = like_count
        self.dislike_count = dislike_count
        self.comment_count = comment_count
        self.tmdb_id = tmdb_id
        self.similarity_score = similarity_score

    def set_similarity_score(self, score):
        self.similarity_score = score

    def __lt__(self, compared_video):
        return self.similarity_score < compared_video.similarity_score