import os
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *


class Fenster(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initMe()
        
    def initMe(self):
        self.p = None
        self.image_path = "git/F-E_Projekt_Montage/Testfoto.jpg"
        self.new_image_path = " "
        self.folder = "git/F-E_Projekt_Montage"
        self.plan = " "
        self.assembly_progress = "Pyramide Schritt 3"
        self.text = QLabel('Der letzte beendete Montageschritt ist: ', self)
        self.text.setGeometry(50,700, 1000, 50)
        
        # creating label
        self.label = QLabel(self)
        self.label.move(50,100)
        # loading image
        self.pixmap = QPixmap('P_Step3_front_template.png')
        # adding image to label
        self.label.setPixmap(self.pixmap)
        # Optional, resize label to image size
        self.label.resize(self.pixmap.width(),
                        self.pixmap.height())

        # shows an image of the selected construction plan
        self.label_plan = QLabel(self)
        self.label_plan.move(350,100)
        # loading image
        self.pixmap_plan = QPixmap('Pyramide_zugeschnitten.jpg')
        # adding image to label
        self.label_plan.setPixmap(self.pixmap_plan)
        self.label_plan.setGeometry(1400, 100, 300, 300)

        # creates an selection field for the construction plan
        self.construction_plan = QComboBox(self)
        self.construction_plan.setGeometry(1400,450, 300, 100)
        self.construction_plan.addItem('Pyramide')
        self.construction_plan.addItem('Burg')
        self.construction_plan.addItem('Brücke')
        self.construction_plan.currentIndexChanged.connect(self.select_construct_plan)

        # creates a button to to show the current progress step
        button = QPushButton('Fortschritt anzeigen', self)
        button.setGeometry(1400, 600, 300, 100)
        #button.move(700, 50)
        button.setToolTip('Zeigt den aktuellen <b>Montageschritt</b> für den ausgewählten Bauplan an')
        button.clicked.connect(self.show_progress)

        # creates a button to capture an image
        button = QPushButton('Foto aufnehmen', self)
        button.setGeometry(1400, 750, 300, 100)
        #button.move(700, 50)
        button.setToolTip('<b>Foto</b> aufnehmen')
        button.clicked.connect(self.start_process)
        self.text2 = QPlainTextEdit()
        self.text2.setReadOnly(True)
        self.text2.setGeometry(1700, 500, 200, 200)

        # creating action for closing the window
        exitMe = QAction('&Exit', self)
        exitMe.setShortcut('Ctrl+E')
        exitMe.triggered.connect(self.close)

        select_pyramid = QAction('&Pyramide auswählen', self)
        select_pyramid.setShortcut('Ctrl+P')
        select_pyramid.triggered.connect(self.close)

        select_castle = QAction('&Burg auswählen', self)
        select_castle.setShortcut('Ctrl+H')
        select_castle.triggered.connect(self.close)

        select_bridge = QAction('&Brücke auswählen', self)
        select_bridge.setShortcut('Ctrl+B')
        select_bridge.triggered.connect(self.close)

        # creating action for capturing an image
        capture_picture = QAction('&Foto aufnehmen', self)
        capture_picture.setShortcut('Ctrl+F')
        capture_picture.triggered.connect(self.start_process)

        # creating action for showing the progress
        show_progress = QAction('&Fortschritt anzeigen', self)
        show_progress.setShortcut('Ctrl+A')
        show_progress.triggered.connect(self.show_progress)

        # creating action for showing information
        instruction = QAction('&Informationen', self)
        instruction.setShortcut('Ctrl+I')
        instruction.triggered.connect(self.show_instructions)

        # creating action for changing save folder
        change_folder = QAction("&Speicherort ändern",self)
        change_folder.triggered.connect(self.change_folder)
        change_folder.setShortcut('Ctrl+S')

        # creates a menu bar
        menuBar = self.menuBar()
        menu = menuBar.addMenu('&Menu')
        construct_plan_menu = menu.addMenu("Bauplan auswählen")
        construct_plan_menu.addAction(select_pyramid)
        construct_plan_menu.addAction(select_castle)
        construct_plan_menu.addAction(select_bridge)
        menu.addAction(capture_picture)
        menu.addAction(show_progress)
        menu.addAction(change_folder)
        menu.addAction(instruction)
        menu.addAction(exitMe)

        self.statusBar().showMessage('Montageinspektion von Stephan Klotz, Jennifer Sissi Lange, Sophia Reiter')

        self.setGeometry(100, 100, 1750, 1000)
        self.setWindowTitle('Montageinspektion')
        self.show()

    def change_folder(self):
        # open the dialog to select path
        path = QFileDialog.getExistingDirectory(self, 
                                                "Picture Location", "")
        # if path is selected
        if path:
            # update the path
            self.folder = path

    def select_construct_plan(self, i):
        # hier wird später ausgewählt, dass Templatebilder des jeweilgen Bauplans getestet werden
        if self.construction_plan.currentText() == 'Pyramide':
            pixmap_plan = QPixmap('Pyramide_zugeschnitten.jpg')
            self.plan = "Pyramide"
            self.label_plan.setPixmap(pixmap_plan)
        if self.construction_plan.currentText() == 'Burg':
            pixmap_plan = QPixmap('Burg_zugeschnitten.jpg')
            self.plan = "Burg"
            self.label_plan.setPixmap(pixmap_plan)
        if self.construction_plan.currentText() == 'Brücke': 
            pixmap_plan = QPixmap('Bruecke_zugeschnitten.jpg')
            self.label_plan.setPixmap(pixmap_plan)
            self.plan = "Bruecke"
        else:
            pass
        #self.label.setPixmap(pixmap_plan)
        print(self.construction_plan.currentText())

    def message(self, s):
        self.text2.appendPlainText(s)

    @pyqtSlot()    
    def start_process(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            self.p.start("python3", ['git/F-E_Projekt_Montage/capture_picture.py']) #selects file to be executed
        text, ok = QInputDialog.getText(self, 'Speicherort', 'Bitte Bezeichung für Foto eingeben:') # entering the file name
        self.new_image_path = text
        new_image_path = os.rename(self.image_path, self.folder +"/"+ self.new_image_path + ".jpg") #changes path of the file
        
    def process_finished(self):
        self.message("Process finished.")
        self.p = None

    def show_progress(self, event):
        #progress = request_progress()
        self.text.setText('Der letzte beendete Montageschritt ist: ' + self.plan + self.assembly_progress)
        if self.new_image_path != " ":
            self.change_image()

    def change_image(self, *args):
        # shows the captured image
        pixmap = QPixmap(self.folder +"/"+ self.new_image_path + ".jpg")
        self.label.setPixmap(pixmap) 
      

    def show_instructions(self):
        info = QMessageBox()
        info.setWindowTitle("Anleitung")
        info.setText("This is the main text!")
        x = info.exec_()  # this will show a messagebox 

    # def request_progress(self):
    #     self. assembly_progress = importierte_methode()

 
if __name__ == '__main__':        

    app = QApplication(sys.argv)
    w = Fenster()

    sys.exit(app.exec_())
