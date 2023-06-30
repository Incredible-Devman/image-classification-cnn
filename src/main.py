import sys
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, pyqtSlot
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np

class MainWindow(QWidget):
    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        loadUi("widget.ui", self)
        self.createSignals()
        self.imageUrl = ''
        self.saved_model = tf.keras.models.load_model('cats_and_dogs.h5')
        print('load model succesfully.')
        
    def createSignals(self):
        self.recogBtn.clicked.connect(self.recogAnimal)
        self.loadBtn.clicked.connect(self.loadImage)

    @pyqtSlot()
    def loadImage(self):
        #load your image of cat or dog for recognization
        d = QFileDialog(self, 'Select an image file')
        d.setNameFilter("*.*")
        if d.exec():
            self.imageUrl = d.selectedFiles()[0]
            pixmap = QPixmap()
            pixmap.load(self.imageUrl)
            self.previewImg.setPixmap(pixmap.scaled(400, 290))

    @pyqtSlot()
    def recogAnimal(self):
        if self.imageUrl == '':
            QMessageBox.warning(self, "Warning", "You should select an image to recognize a dog or a cat.")
        else:
            img_obj = Image.open(self.imageUrl).resize((150, 150))
            image_np = np.array(img_obj) / 256.0
            image_np = np.expand_dims(image_np, 0)
            ret = self.saved_model.predict(image_np)
            if ret[0][0] > 0.0:
                QMessageBox.about(self, "Result", "The animal is a dog.")
            else:
                QMessageBox.about(self, "Result", "The animal is a cat.")

app = QApplication(sys.argv)
widget = MainWindow()
widget.show()
sys.exit(app.exec())
