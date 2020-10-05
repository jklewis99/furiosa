from moviepy.editor import VideoFileClip
from classes.clip import FuriosaVideoClip
import time
# v1 = v2 = VideoFileClip(r'C:\Users\jklew\Videos\flip.mp4')
# v2 = v2.set_end(5.0)
# v1.preview()

def main():
    path = r"C:\Users\jklew\Videos\ClimbVideo.mp4"
    start = time.time()
    clip = FuriosaVideoClip(path=path)
    print(time.time()-start)
    start = time.time()
    clip.trim("front", 230.0)
    print(clip.get_duration())
    print(time.time()-start)

if __name__ == "__main__":
    main()