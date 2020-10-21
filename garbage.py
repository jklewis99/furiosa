import heapq
from youtube_video import YouTubeVideo
a = YouTubeVideo("","","","","","",0,0,0,0,12)
b = YouTubeVideo("","","","","","",0,0,0,0,2)
c = YouTubeVideo("","","","","","",0,0,0,0,3)
d = YouTubeVideo("","","","","","",0,0,0,0,5)

h = []
heapq.heappush(h, a)
heapq.heappush(h, b)
heapq.heappush(h, c)
heapq.heappush(h, d)
print(h[0].similarity_score)