import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QEvent, QObject, Qt, QRect


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 500
        self.top = 500
        self.width = 320
        self.height = 240
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        self.label = QLabel(self)
        pixmap = QPixmap('logo.jpg')
        pixmap = pixmap.scaled(160, 120, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)
        self.resize(self.width, self.height)

        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pixmap = QPixmap('logo.jpg')
            pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

        if event.button() == Qt.RightButton:
            pixmap = QPixmap('logo.jpg')
            self.showFullScreen()
            # self.resize(pixmap.width(), pixmap.height())
            # self.label.show()

    # def eventFilter(self, obj: QLabel, event: QMouseEvent) -> bool:
    #     if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
        #     if self.parent.mediaAvailable and self.isEnabled():
        #         newpos = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x() - self.offset,
        #                                                 self.width() - (self.offset * 2))
        #         self.setValue(newpos)
        #         self.parent.setPosition(newpos)
        #         self.parent.parent.mousePressEvent(event)
        # return super(VideoSlider, self).eventFilter(obj, event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
