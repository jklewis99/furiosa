import numpy as np
import cv2
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QSizePolicy, QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage                                
from PyQt5.QtCore import QCoreApplication, QRectF, Qt, QRect, QMetaObject                                                                                          
import numpy as np                                                     
import sys
import time
from moviepy.editor import VideoFileClip

class UI_Main(object):
    def setup_UI(self, MainWindow):
        MainWindow.setObjectName("MainWIndow")
        MainWindow.resize(800, 600)
        self.video = VideoFileClip(r'C:\Users\jklew\Videos\Music\Fractalia.MP4')
        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        self.pixmap_label = QLabel(self.centralWidget)
        self.pixmap_label.setGeometry(QRect(0, 0, 841, 511))
        self.pixmap_label.setText("")
        self.pixmap_label
        im_np = self.video.get_frame(0)
        print(im_np.dtype)
        # im_np = np.ones((1800,2880,3),dtype='uint8')                                                                                                                                                                                  
        # im_np = np.transpose(im_np, (1,0,2))
        # im_np = np.transpose(im_np,(1,0,2)).copy()                                                                                                                                                                         
        # im_np = im_np.copy()                                                                                             
        qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],                                                                                                                                                 
                        QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        # pixmap = pixmap.scaled(640,400, Qt.KeepAspectRatio)
        self.pixmap_label.setPixmap(pixmap)
        self.pixmap_label.setScaledContents(True)
        self.pixmap_label.setObjectName("Photo")
        MainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUI(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUI(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))  



class Test(QMainWindow):                                                                                                                                                                                       

    def __init__(self):                                                                                                                                                                                        
        super().__init__()
        self.video = VideoFileClip(r'C:\Users\jklew\Videos\Music\Fractalia.MP4')
        self.an_image = QImage(r"C:\Users\jklew\OneDrive\Pictures\madmax.jpg")
        self.initUI()

    def initUI(self):                                                                                                                                                                                          
        self.setGeometry(30,40, 1000, 1000)
        self.centralWidget = QWidget(self)
        self.centralWidget.setObjectName("centralWidget")

        self.pixmap_label = QLabel(self.centralWidget)
        # pixmap_label.setGeometry(QRect(30, 40, 640, 400))
        # pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # self.pixmap_label.resize(800,800)
        # self.pixmap_label.setAlignment(Qt.AlignCenter)

        im_np = self.video.get_frame(0)
        print(im_np.dtype)
        # im_np = np.ones((1800,2880,3),dtype='uint8')                                                                                                                                                                                  
        # im_np = np.transpose(im_np, (1,0,2))
        # im_np = np.transpose(im_np,(1,0,2)).copy()                                                                                                                                                                         
        # im_np = im_np.copy()                                                                                             
        qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],                                                                                                                                                 
                        QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        # pixmap = pixmap.scaled(640,400, Qt.KeepAspectRatio)
        # self.pixmap_label.setPixmap(pixmap)
        


        self.play_button = QPushButton(self.centralWidget)
        self.play_button.setGeometry(QRect(20, 810, 50, 50))
        self.play_button.setObjectName("PLAY")
        self.play_button.clicked.connect(self.display_clip2)

        self.setCentralWidget(self.centralWidget)
        self.retranslateUI()                                                                                                                                                                      
  
    def retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.play_button.setText(QCoreApplication.translate("MainWindow", "PLAY"))

    def paintEvent(self, event):
        pen = QPen()
        pen.setWidth(5)
        print(self.centralWidget.rect())
        painter = QPainter(self.pixmap_label)
        painter.drawImage(self.pixmap_label.rect(), self.an_image)
        painter.setPen(pen)
        painter.drawEllipse(300, 300, 500, 500)

    def imdisplay(self, img_array, screen=None):
        # fill the widget with the image array
        # TODO: Qt widget
        screen = self.pixmap_label
        qimage = QImage(img_array, img_array.shape[1], img_array.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(640,400, Qt.KeepAspectRatio)
        screen.setPixmap(pixmap)
    
    def new_meth(self):
        # TODO: implement QWidget.repaint() event
        print()

    def display_clip2(self):
        img = self.video.get_frame(70)
        self.imdisplay(img)

    def display_clip(self, clip, fps=1, audio=False, audio_fps=22050, audio_buffersize=3000,
            audio_nbytes=2):
        """ 
        Displays the clip in a window, at the given frames per second
        (of movie) rate. It will avoid that the clip be played faster
        than normal, but it cannot avoid the clip to be played slower
        than normal if the computations are complex. In this case, try
        reducing the ``fps``.
        
        Parameters
        ------------
        
        fps
        Number of frames per seconds in the displayed video.
            
        audio
        ``True`` (default) if you want the clip's audio be played during
        the preview.
            
        audio_fps
        The frames per second to use when generating the audio sound.
        
        fullscreen
        ``True`` if you want the preview to be displayed fullscreen.
        
        """
        
        # compute and splash the first image
        # TODO: change pgame to a widget in Qt
        # screen = pg.display.set_mode(clip.size, flags)
        
        audio = audio and (clip.audio is not None)
        
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
        img = clip.get_frame(0)

        self.imdisplay(img)
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

def main():                                                                                                                                                                                                    
    app = QApplication(sys.argv)                                                                                                                                                                               
    win = Test()
    win.show()                                                                                                                                                                                          
    # app = QApplication(sys.argv)
    # MainWindow = QMainWindow()
    # ui = UI_Main()
    # ui.setup_UI(MainWindow)
    # MainWindow.show()
    sys.exit(app.exec_())  

if __name__ == "__main__":
    main()


def mainish():
    img = cv2.imread('test.png')[:,:,::1]/255. 
    height, width, channels = img.shape
    bytesPerLine = 3 * width
    qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
    imgDown = cv2.pyrDown(img)
    imgDown = np.float32(imgDown)        
    cvRGBImg = cv2.cvtColor(imgDown, cv2.cv.CV_BGR2RGB)
    qimg = QtGui.QImage(cvRGBImg.data,cvRGBImg.shape[1], cvRGBImg.shape[0], QtGui.QImage.Format_RGB888)
    pixmap01 = QtGui.QPixmap.fromImage(qimg)
    self.image01TopTxt = QtGui.QLabel('window',self)
    self.imageLable01 = QtGui.QLabel(self)
    self.imageLable01.setPixmap(pixmap01)
