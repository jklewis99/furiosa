# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2

class Ui_MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.setWindowTitle("Furiosa")
        self.setup_ui()
        self.setup_media()

    def setup_ui(self):
        # basically all the buttons in the menu bar
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1255, 25))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.menuOpen_Recent = QtWidgets.QMenu(self.menuFile)
        self.menuOpen_Recent.setObjectName("menuOpen_Recent")

        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")

        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")

        self.setStatusBar(self.statusbar)

        self.actionNew = QtWidgets.QAction(self)
        self.actionNew.setObjectName("actionNew")

        self.actionImport = QtWidgets.QAction(self)
        self.actionImport.setObjectName("actionImport")
        self.actionImport.triggered.connect(self.import_file)

        self.actionOpen = QtWidgets.QAction(self)
        self.actionOpen.setObjectName("actionOpen")

        self.actionMore = QtWidgets.QAction(self)
        self.actionMore.setObjectName("actionMore")

        self.actionSave = QtWidgets.QAction(self)
        self.actionSave.setObjectName("actionSave")

        self.actionSave_As = QtWidgets.QAction(self)
        self.actionSave_As.setObjectName("actionSave_As")

        self.actionExit = QtWidgets.QAction(self)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exit_call)

        self.actionUndo = QtWidgets.QAction(self)
        self.actionUndo.setObjectName("actionUndo")

        self.actionRedo = QtWidgets.QAction(self)
        self.actionRedo.setObjectName("actionRedo")

        self.actionCut = QtWidgets.QAction(self)
        self.actionCut.setObjectName("actionCut")

        self.actionCopy = QtWidgets.QAction(self)
        self.actionCopy.setObjectName("actionCopy")

        self.actionPaste = QtWidgets.QAction(self)
        self.actionPaste.setObjectName("actionPaste")

        self.menuOpen_Recent.addSeparator()
        self.menuOpen_Recent.addAction(self.actionMore)

        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionImport)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.menuOpen_Recent.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)

        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionCut)
        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addAction(self.actionPaste)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.setup_menu_bar()
        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_menu_bar(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Furiosa"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuOpen_Recent.setTitle(_translate("MainWindow", "Open Recent"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionNew.setStatusTip(_translate("MainWindow", "Create a new project"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionImport.setText(_translate("MainWindow", "Import"))
        self.actionImport.setStatusTip(_translate("MainWindow", "Import a file to the project"))
        self.actionImport.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionOpen.setText(_translate("MainWindow", "Open Project"))
        self.actionOpen.setStatusTip(_translate("MainWindow", "Open Project"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionMore.setText(_translate("MainWindow", "More..."))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save Project"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Exit the program"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionUndo.setStatusTip(_translate("MainWindow", "Undo"))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionRedo.setStatusTip(_translate("MainWindow", "Redo"))
        self.actionRedo.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actionCut.setText(_translate("MainWindow", "Cut"))
        self.actionCut.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.actionCopy.setText(_translate("MainWindow", "Copy"))
        self.actionCopy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionPaste.setText(_translate("MainWindow", "Paste"))
        self.actionPaste.setShortcut(_translate("MainWindow", "Ctrl+V"))

    def init_buttons(self):
        # how to set up a play button:
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play) # call the play method

    def init_sliders(self):
        # slider for playback position
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

    def init_error_response(self):
        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)  

    def setup_media(self):
        # setup for the video playback
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        project_video_widget = QVideoWidget()
        
        self.init_buttons()
        self.init_sliders()
        self.init_error_response()    

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)
        grid = QGridLayout()
        wid.setLayout(grid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        # controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        # 2 video panels??
        clip_video_widget = QLabel()
        clip_video_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        clip_video_widget.resize(1000, 1000)
        # clip_video_widget.setAlignment(Qt.AlignCenter)
        # clip_video_widget.setPixmap(QtGui.QPixmap("C:/Users/jklew/OneDrive/Pictures/madmax.jpg"))
        # clip_video_widget.setScaledContents(True)
        im_np = cv2.imread("C:/Users/jklew/OneDrive/Pictures/madmax.jpg")
        clip_frame = QImage(im_np, im_np.shape[1], im_np.shape[0], QImage.Format_RGB888)
        clip_pixmap = QPixmap(clip_frame)
        clip_pixmap = clip_pixmap.scaled(1000, 1000, Qt.KeepAspectRatio)
        clip_video_widget.setPixmap(clip_pixmap)
        clip_video_widget.setMinimumWidth(700) # only way to see it on window?
        # clip_video_widget.setMaximumSize(1920//2, 1080//2)
        left_layout = QVBoxLayout()
        left_layout.addWidget(clip_video_widget)
        # left_layout.addLayout(controlLayout)

        right_layout = QVBoxLayout()
        # right_layout.addWidget(clip_video_widget)
        right_layout.addWidget(project_video_widget)
        right_layout.addLayout(controlLayout)
        right_layout.addWidget(self.errorLabel)

        # layout = QHBoxLayout()
        # layout.addLayout(left_layout)
        # layout.addLayout(right_layout)
        
        

        # Set widget to contain window contents
        # grid.addWidget(clip_video_widget, 0, 0)
        grid.addLayout(left_layout, 0, 0)
        grid.addLayout(right_layout, 0, 1)
        # wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(project_video_widget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def import_file(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def exit_call(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.showMaximized()
    sys.exit(app.exec_())