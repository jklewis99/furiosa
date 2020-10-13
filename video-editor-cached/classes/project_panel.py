import sys
from io import StringIO

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject
from PyQt5.QtGui import QCloseEvent, QShowEvent, QTextCursor, QTextOption
from PyQt5.QtWidgets import qApp, QDialog, QDialogButtonBox, QStyleFactory, QTextEdit, QVBoxLayout

# documentation from vidcutter... let's see if this works
class VideoPanel(QTextEdit):
    def __init__(self, parent=None):
        super(VideoPanel, self).__init__(parent)
        self._buffer = StringIO()
        self.setReadOnly(True)
        self.setWordWrapMode(QTextOption.WordWrap)
        self.setStyleSheet('QTextEdit { font-family:monospace; font-size:%s; }'
                           % ('10pt' if sys.platform == 'darwin' else '8pt'))
    def __getattr__(self, item):
        return getattr(self._buffer, item)      

class ConsoleWidget(QDialog):
    def __init__(self, parent=None, flags=Qt.Dialog | Qt.WindowCloseButtonHint):
        super(ConsoleWidget, self).__init__(parent, flags)
        self.parent = parent
        self.edit = VideoPanel(self)
        buttons = QDialogButtonBox()
        buttons.setCenterButtons(True)
        clearButton = buttons.addButton('Clear', QDialogButtonBox.ResetRole)
        clearButton.clicked.connect(self.edit.clear)
        closeButton = buttons.addButton(QDialogButtonBox.Close)
        closeButton.clicked.connect(self.close)
        closeButton.setDefault(True)
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setWindowTitle('{0} Console'.format(qApp.applicationName()))
        self.setWindowModality(Qt.NonModal)

    def showEvent(self, event: QShowEvent):
        self.parent.consoleLogger.flush()
        super(ConsoleWidget, self).showEvent(event)

    def closeEvent(self, event: QCloseEvent):
        self.parent.cutter.consoleButton.setChecked(False)
        super(ConsoleWidget, self).closeEvent(event)