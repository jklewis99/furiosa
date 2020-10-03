# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import time
import numpy as np
from moviepy.editor import VideoFileClip
from read_video import count_frames
from numba import vectorize, jit, cuda 

class PhotoBoothApp:
    def __init__(self, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        # self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!",
        command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
        pady=10)
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.display_clip, args=())
        self.thread.start()
        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=300)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    # @vectorize(['float64(float64)'], target="cuda") # I want to use gpu :'(
    def display_clip(self, fps=30, audio=False, audio_fps=22050, audio_buffersize=3000,
    audio_nbytes=2):

        # audio = audio and (clip.audio is not None)

        # if audio:
        #     # the sound will be played in parrallel. We are not
        #     # paralellizing it on different CPUs because it seems that
        #     # pygame and openCV already use several cpus it seems.

        #     # two synchro-flags to tell whether audio and video are ready
        #     videoFlag = threading.Event()
        #     audioFlag = threading.Event()
        #     # launch the thread
        #     audiothread = threading.Thread(target=clip.audio.preview,
        #                                 args=(audio_fps,
        #                                         audio_buffersize,
        #                                         audio_nbytes,
        #                                         audioFlag, videoFlag))
        #     audiothread.start()
        path = r"C:\Users\jklew\Videos\ClimbVideo.mp4"
        num_frames = count_frames(path)
        clip = VideoFileClip(path)
        # print(num_frames)
        # print(clip.duration) # time, in seconds, of clip
        true_fps = num_frames / clip.duration

        # print("FPS:", true_fps)
        # img = clip.get_frame(0)
        
        # self.imdisplay(img)
        # if audio:  # synchronize with audio
        #     videoFlag.set()  # say to the audio: video is ready
        #     audioFlag.wait()  # wait for the audio to be ready

        result = []

        # for t in np.arange(1.0 / fps, clip.duration-.001, 1.0 / fps):
        t = 0
        t0 = time.time()
        while t < num_frames:
            self.frame = clip.get_frame(t*1.0/fps)
            self.frame = imutils.resize(self.frame, width=300)

            image = Image.fromarray(self.frame)
            image = ImageTk.PhotoImage(image)
            
            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tki.Label(image=image)
                self.panel.image = image
                self.panel.pack(side="left", padx=10, pady=10)
            else:
                self.panel.configure(image=image)
                self.panel.image = image
            
            time.sleep(max(1.0/fps-time.time()-t0, 0)) # loop at framerate specified
            t += round(true_fps/fps)
            t0 = time.time()

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        # save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        exit()
        self.stopEvent.set()
        self.root.quit()