import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageClip
import cv2

# TODO: import fps of project
PROJECT_FPS = 24
# TODO: create fucntion that separates audio data
# and video data to create thier respective clips

class Clip():
    '''
    top level class for audio and video clips
    '''
    def __init__(self):
        # self.__duration = None
        self.media = None

    # def get_duration(self):
    #     return self.__duration


class FuriosaVideoClip(Clip):
    '''
    class of furiosa video clip object

    can be created from an imported path OR from a
    moviepy VideoClip object
    '''
    def __init__(self, path=None, clip=None, start=0, end=None):
        super(FuriosaVideoClip, self).__init__()
        self.path_extension = None
        self.__duration = end
        self.end = 0
        self.start = 0
        self.clip_name = None
        self.video_clip = None
        self.__original = None
        self.create_video_clip(path=path, clip=clip, start=0, end=end)

    def create_video_clip(self, path=None, clip=None, start=0, end=None):
        '''
        top-level method to instantiate this object's attributes
        '''
        if path:
            self.video_clip_from_path(path)
        elif clip:
            self.video_clip_from_clip(clip, start, end)
        else:
            print('ERROR: Must specify a path or reference to moviepy VideoClip')
        self.__set_duration()
        
    def video_clip_from_path(self, path):
        '''
        create a video clip out of the file path specified

        Parameters
        ----------
        path:
            String of path to file with supported extension
        '''
        self.path_extension = path.split(".")[-1]
        if self.path_extension in ['jpg', 'png', 'jpeg']:
            self.video_clip = self.__original = ImageClip(path, fps=PROJECT_FPS, duration=self.__duration)
            
        elif self.path_extension in ['mp4', 'mov', 'avi', 'gif']:
            self.video_clip = self.__original = VideoFileClip(path)
        else:
            print('ERROR: File Specified could not be found or the extension \
                is not currently supported.')
        self.end = self.video_clip.duration

    def video_clip_from_clip(self, clip, start, end):
        '''
        create a video clip out of a reference to moviepy VideoClip

        Parameters
        ----------
        clip:
            moviepy VideoFileClip object

        start:
            time (float) in reference to clip from which to start new clip

        end:
            time (float) in refernce to clip at which to end new clip
        '''
        self.__original = clip
        self.video_clip = clip.subclip(start, end)
        self.start = start
        self.end = end

    def trim(self, trim_from, time):
        '''
        method to cut the ends of a clip

        Parameters
        ----------
        trim_from: 
            the end from which the trimming will occur, but it must be 
            "front" or "back"

        time: time stamp of when video will now start/end

        Return
        ------
        the updated copy of the new clip
        '''

        if trim_from == "front":
            self.start = time
            self.video_clip = self.video_clip.set_start(time)
        elif trim_from == "back":
            self.end = time
            self.video_clip = self.video_clip.set_end(time)
        print(self.video_clip.duration)
        self.__set_duration()

        return self.video_clip # I may not need to return

    # TODO: this method may be better outside of the current clip
    def split(self, split_point=None):
        '''
        method to split current video clip into 2 video clips,
            but preserves the original clip in both. This method
            will make the current clip the first subclip and will
            return a reference to teh new second subclip

        Parameters
        ----------
        split_point: defaults to the midpoint, but will accept
            a floating point number specifying time at which 
            to split

        Return
        ------
        tuple of references to self and new clip
        '''
        if not split_point:
            split_point = self.end / 2.0
        
        previous_end = self.end
        subclip1 = self.trim("front", split_point) # also is self
        # create a reference to a new FuriosaVideoClip object
        subclip2 = FuriosaVideoClip(clip=self.__original, start=split_point, end=previous_end)
        self.__set_duration()
        return subclip1, subclip2

    # TODO: iterable effects, though this is not a recommended method
    def apply_filter(self):
        all_frames = [frame for frame in self.video_clip.iter_frames()]

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
    
    def reset(self):
        self.video = self.__original

    # TODO: getters
    def get_original(self):
        return self.__original

    def get_duration(self):
        return self.__duration

    # TODO: private methods
    def __set_duration(self):
        self.__duration = self.end - self.start


class AudioClip(Clip):
    def __init__(self):
        super(AudioClip, self).__init__()
        self.audio_name = None
    # TODO: User moviepy to get audio array from video

