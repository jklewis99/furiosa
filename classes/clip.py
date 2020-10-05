import numpy as np
from moviepy.editor import VideoFileClip
import cv2

# TODO: create fucntion that separates audio data
# and video data to create thier respective clips

class Clip():
    '''
    top level class for audio and video clips
    '''
    def __init__(self):
        self.__length = None
        self.media = None

    def get_length(self):
        return self.__length


class VideoClip(Clip):
    def __init__(self, path):
        super(VideoClip, self).__init__()
        self.video_name = None
        self.frames = None

    # TODO: Transfer VideoClip object methods to moviepy objects and methods
    def extract_frames(self, path):
        video = cv2.VideoCapture(path)
        height, width = self.get_dimensions(path)
        num_frames = self.count_frames(path)

        # create numpy array for all the frames (convert video to array)
        video = self.numpy_video(path, num_frames, height, width)

        return video

    def numpy_video(self, path, num_frames, height, width, num_channels=3):
        # TODO: address failure on high resolution videos of long duration
        video = cv2.VideoCapture(path)
        video_array = np.empty((num_frames, height, width, num_channels), np.dtype('uint8'))
        frame_idx = 0

        while video.isOpened():
            ret, frame = video.read()
            if ret:
                video_array[frame_idx] = frame
            else:
                print("1 ERROR: Error reading video", frame_idx)
                break
            frame_idx += 1
        video.release()
        return video_array

    def get_dimensions(self, path):
        video = cv2.VideoCapture(path)
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
        return h, w

    def count_frames(self, path):
        video = cv2.VideoCapture(path)
        num_frames = 0
        
        try:
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            num_frames = self.manual_count_frames(video)
        video.release()
        return num_frames

    def manual_count_frames(self, video):
        num_frames = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("2 ERROR: Error reading video.")
            num_frames += 1

        return num_frames

class AudioClip(Clip):
    def __init__(self):
        super(AudioClip, self).__init__()
        self.audio_name = None
    # TODO: User moviepy to get audio array from video

def main():
    path = "path/to/video"
    clip = VideoClip(path)
    print(clip.get_length())

if __name__ == "__main__":
    main()