### STILL DOESN'T WORK

import numpy as np
import cv2
from PyQt5.QtWidgets import QMainWindow, QLabel, QOpenGLWidget, QPushButton, QSizePolicy, QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage                                
from PyQt5.QtCore import QCoreApplication, QRectF, Qt, QRect, QMetaObject                                                                                          
import numpy as np                                                     
import sys
import time
from moviepy.editor import VideoFileClip

class VideoPanel(QOpenGLWidget):

    def __init__(self, parent=None, file=None, video_file=None):
        super(VideoPanel, self).__init__(parent)
        self.parent = parent
        self.image = None
        self.video = VideoFileClip(video_file)

    # def paintEvent(self, event):
    #     pen = QPen()
    #     pen.setWidth(5)
    #     painter = QPainter(self)
    #     painter.drawImage(self, self.image)
    #     painter.setPen(pen)
    #     painter.drawEllipse(300, 300, 500, 500)

    def display_clip(self, fps=60, audio=False, audio_fps=22050, audio_buffersize=3000,
            audio_nbytes=2):
        
        # audio = audio and (clip.audio is not None)
        
        if audio:
            # the sound will be played in parrallel. We are not
            # paralellizing it on different CPUs because it seems that
            # pygame and openCV already use several cpus it seems.
            
            # two synchro-flags to tell whether audio and video are ready
            videoFlag = threading.Event()
            audioFlag = threading.Event()
            # launch the thread
            audiothread = threading.Thread(target=clip.audio.preview,
                                        args=(audio_fps,
                                                audio_buffersize,
                                                audio_nbytes,
                                                audioFlag, videoFlag))
            audiothread.start()
        clip = self.video
        # img = clip.get_frame(0)

        # self.imdisplay(img)
        if audio:  # synchronize with audio
            videoFlag.set()  # say to the audio: video is ready
            audioFlag.wait()  # wait for the audio to be ready
        
        result = []
        
        t0 = time.time()
        for t in np.arange(1.0 / fps, clip.duration-.001, 1.0 / fps):
            
            img = clip.get_frame(t)
            print(img.shape)
            t1 = time.time()
            time.sleep(max(0, t - (t1-t0))) # loop at framerate specified
            self.imdisplay(img) #, screen)
    
    def imdisplay(self, img_array):
        # fill the widget with the image array
        # TODO: Qt widget
        self.image = QImage(img_array, img_array.shape[1], img_array.shape[0], QImage.Format_RGB888)
        self.repaint()


class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.video = VideoFileClip(r'C:\Users\jklew\Videos\Music\Fractalia.MP4')
        im_np = self.video.get_frame(0)
        self.image = QImage(im_np, im_np.shape[1], im_np.shape[0],                                                                                                                                                 
                        QImage.Format_RGB888)
    

    
def main():
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    demo.display_clip()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()