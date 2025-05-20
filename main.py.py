# main.py
import sys
import subprocess
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QApplication, QMainWindow
from keras.models import load_model
from PIL import Image
import numpy as np

class_labels = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Classifier")
        self.setGeometry(100, 100, 800, 600)
        self.setupUI()

    def setupUI(self):
        # Browse button
        self.browseBtn = QPushButton("Predict", self)
        self.browseBtn.setGeometry(160, 370, 151, 51)
        self.browseBtn.clicked.connect(self.load_image)

        # Train button
        self.trainBtn = QPushButton("Train", self)
        self.trainBtn.setGeometry(350, 370, 151, 51)
        self.trainBtn.clicked.connect(self.train_model)

        # Image display
        self.imageLbl = QLabel(self)
        self.imageLbl.setGeometry(200, 80, 361, 261)
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")

        # Result label
        self.resultLbl = QLabel("Prediction: ", self)
        self.resultLbl.setGeometry(200, 450, 500, 50)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.resultLbl.setFont(font)

    def train_model(self):
        subprocess.Popen(["cmd", "/c", "python train.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            pixmap = QtGui.QPixmap(file_path)
            self.imageLbl.setPixmap(pixmap.scaled(self.imageLbl.size(), QtCore.Qt.KeepAspectRatio))
            self.predict(file_path)

    def predict(self, image_path):
        try:
            model = load_model("traffic_sign_model.h5")
            image = Image.open(image_path).resize((30, 30))
            image = np.expand_dims(np.array(image), axis=0)
            pred = model.predict(image)
            class_id = np.argmax(pred)
            self.resultLbl.setText(f"Prediction: {class_labels[class_id]}")
        except Exception as e:
            self.resultLbl.setText("Error loading model or image.")
            print(e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec_())
