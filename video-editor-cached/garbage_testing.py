from classes.clip import FuriosaVideoClip
import time
# v1 = v2 = VideoFileClip(r'C:\Users\jklew\Videos\flip.mp4')
# v2 = v2.set_end(5.0)
# v1.preview()
import numpy as np                                                     
import sys
import time
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication, QWidget
import threading
from moviepy.editor import VideoFileClip

class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.video = VideoFileClip(r'C:\Users\jklew\Videos\Music\Fractalia.MP4') # I am using a real video
        im_np = self.video.get_frame(0)
        self.image = QImage(im_np, im_np.shape[1], im_np.shape[0],                                                                                                                                                 
                        QImage.Format_RGB888)
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.display_clip, args=())
        self.thread.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)

    def display_clip(self, fps=60):
        clip = self.video
        img = clip.get_frame(0) # returns numpy array of frame at time 0

        t0 = time.time()

        for t in np.arange(1.0 / fps, clip.duration-.001, 1.0 / fps):
            
            img = clip.get_frame(t) # returns numpy array of frame at time t
            # print(img.shape)
            t1 = time.time()
            time.sleep(max(0, t - (t1-t0))) # loop at framerate specified
            self.imdisplay(img) #, screen)
    
    def imdisplay(self, img_array):
        # fill the widget with the image array
        # TODO: Qt widget
        self.image = QImage(img_array, img_array.shape[1], img_array.shape[0], QImage.Format_RGB888)
        self.repaint()

def main1():
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    # demo.display_clip()
    sys.exit(app.exec_())

def main():
    path = r"C:\Users\jklew\Videos\flip.mp4"
    start = time.time()
    clip = FuriosaVideoClip(path=path)
    print(time.time() - start)
    
    # trim testing
    # start = time.time()
    # clip.trim("front", 5.0)
    # print(clip.get_duration())
    # clip.video_clip.preview()
    # print(time.time() - start)

    # split testing
    start = time.time()
    subclip1, subclip2 = clip.split(4.0)
    print(time.time()-start)
    # print(subclip1.get_duration())
    # subclip1.video_clip.preview()
    # print(subclip2.get_duration())
    # subclip2.video_clip.preview()

if __name__ == "__main__":
    main()