import os
import cv2
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from functions import *
#from matrix import *




#TODO Label für die 4 Bilder anlegen, bei Kopfdruck Bilder aktualisieren --> Funktion, die die Pixmap aktualisiert
class Fenster(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initMe()
        
    def initMe(self):
        self.p = None
        #self.image_path = "git/F-E_Projekt_Montage/Testfoto.jpg"
        #self.new_image_path = " "
        #self.folder = "git/F-E_Projekt_Montage"
        #self.plan = " "
        #self.assembly_progress = " Schritt 3"
        self.text = QLabel('Der letzte beendete Montageschritt ist: ', self)
        self.text.setGeometry(0,1600, 1000, 50)
        
        # creating label
        #self.label = QLabel(self)
        #self.label.move(50,100)
        # loading image
        #self.pixmap = QPixmap('/home/sophia/workspace/FuE/P_real_front3.jpg')
        # adding image to label
        #self.label.setPixmap(self.pixmap)
        # Optional, resize label to image size
        #self.label.resize(self.pixmap.width(),
        #                self.pixmap.height())
        
#  QPixmap image("image address");
# h=ui->label->height;
# w=ui->label->width;
# ui->label->setPixmap(image.scaled(w,h,Qt::KeepAspectRatio));
        
        #show captured picture
        self.label_tl = QLabel(self)
        #self.label_tl.move(50, 100)
        self.label_tl.setGeometry(50, 100, 700, 700)
        self.pixmap_tl = QPixmap("/home/sophia/workspace/FuE/Fotos/ids_pyramide/pyramide_crop1.jpg")
        self.label_tl.setPixmap(self.pixmap_tl.scaled(700,700,Qt.KeepAspectRatio))
       
        #self.label_tl.resize(self.pixmap_tl.width(),
        #                self.pixmap_tl.height())
        
        #show cropped picture with circles
        self.label_tr = QLabel(self)
        #self.label_tr.move(550, 100)
        self.label_tr.setGeometry(850, 100, 700, 700)
        self.pixmap_tr = QPixmap("/home/sophia/workspace/FuE/Fotos/ids_pyramide/pyramide_crop1.jpg")
        self.label_tr.setPixmap(self.pixmap_tr.scaled(700,700,Qt.KeepAspectRatio))
        
        #self.label_tr.resize(self.pixmap_tr.width(),
        #                self.pixmap_tr.height())
        
        #show colored matrix
        self.label_bl = QLabel(self)
        #self.label_bl.move(50, 600)
        self.label_bl.setGeometry(50, 850, 700, 700)
        self.pixmap_bl = QPixmap("/home/sophia/workspace/FuE/Fotos/ids_pyramide/pyramide_crop1.jpg")
        self.label_bl.setPixmap(self.pixmap_bl.scaled(700,700,Qt.KeepAspectRatio))
        #self.label_bl.resize(self.pixmap_bl.width(),
        #                self.pixmap_bl.height())
        
        #show cropped and highlighted picture
        self.label_br = QLabel(self)
        #self.label_br.move(550, 600)
        self.label_br.setGeometry(850, 850, 700, 700)
        self.pixmap_br = QPixmap("/home/sophia/workspace/FuE/Fotos/ids_pyramide/pyramide_crop1.jpg")
        self.label_br.setPixmap(self.pixmap_br.scaled(700,700,Qt.KeepAspectRatio))
        #self.label_br.resize(self.pixmap_br.width(),
        #                self.pixmap_br.height())


        # shows an image of the selected construction plan
        self.label_plan = QLabel(self)
        self.label_plan.move(350,100)
        # loading image
        self.pixmap_plan = QPixmap('Pyramide_zugeschnitten.jpg')
        # adding image to label
        self.label_plan.setPixmap(self.pixmap_plan)
        self.label_plan.setGeometry(2200, 100, 300, 300)

        # creates an selection field for the construction plan
        self.construction_plan = QComboBox(self)
        self.construction_plan.setGeometry(2200,450, 300, 100)
        self.construction_plan.addItem('Pyramid')
        self.construction_plan.addItem('Castle')
        self.construction_plan.addItem('Bridge')
        self.construction_plan.currentIndexChanged.connect(self.select_construct_plan)

        # creates a button to to show the current progress step
        button = QPushButton('Show progress', self)
        button.setGeometry(2200, 600, 300, 100)
        #button.move(700, 50)
        button.setToolTip('Shows the current <b>progress</b> for the selected plan')
        #button.clicked.connect(self.show_progress)
        button.clicked.connect(self.detect_progress)

        # creates a button to capture an image
        button = QPushButton('Take photo', self)
        button.setGeometry(2200, 750, 300, 100)
        #button.move(700, 50)
        button.setToolTip('take <b>Photo</b>')
        button.clicked.connect(self.start_process)
        self.text2 = QPlainTextEdit()
        self.text2.setReadOnly(True)
        self.text2.setGeometry(1700, 500, 200, 200)

        # creating action for closing the window
        exitMe = QAction('&Exit', self)
        exitMe.setShortcut('Ctrl+E')
        exitMe.triggered.connect(self.close)

        select_pyramid = QAction('select &Pyramid', self)
        select_pyramid.setShortcut('Ctrl+P')
        select_pyramid.triggered.connect(self.close)

        select_castle = QAction('select &castle', self)
        select_castle.setShortcut('Ctrl+H')
        select_castle.triggered.connect(self.close)

        select_bridge = QAction('select &Bridge', self)
        select_bridge.setShortcut('Ctrl+B')
        select_bridge.triggered.connect(self.close)

        # creating action for capturing an image
        capture_picture = QAction('take &photo', self)
        capture_picture.setShortcut('Ctrl+F')
        capture_picture.triggered.connect(self.start_process)

        # # creating action for showing the progress
        # show_progress = QAction('show &progress', self)
        # show_progress.setShortcut('Ctrl+A')
        # show_progress.triggered.connect(self.show_progress)

        # creating action for showing information
        instruction = QAction('&Information', self)
        instruction.setShortcut('Ctrl+I')
        instruction.triggered.connect(self.show_instructions)

        # creating action for changing save folder
        change_folder = QAction('change &saving location',self)
        change_folder.triggered.connect(self.change_folder)
        change_folder.setShortcut('Ctrl+S')

        # creates a menu bar
        menuBar = self.menuBar()
        menu = menuBar.addMenu('&Menu')
        construct_plan_menu = menu.addMenu('select plan')
        construct_plan_menu.addAction(select_pyramid)
        construct_plan_menu.addAction(select_castle)
        construct_plan_menu.addAction(select_bridge)
        menu.addAction(capture_picture)
        #menu.addAction(show_progress)
        menu.addAction(change_folder)
        menu.addAction(instruction)
        menu.addAction(exitMe)

        self.statusBar().showMessage('Montageinspektion von Stephan Klotz, Jennifer Sissi Lange, Sophia Reiter')

        self.setGeometry(100, 100, 2750, 2000)
        self.setWindowTitle('Assembly inspection')
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
        if self.construction_plan.currentText() == 'Pyramid':
            pixmap_plan = QPixmap('Pyramide_zugeschnitten.jpg')
            self.plan = "Pyramid"
            self.label_plan.setPixmap(pixmap_plan)
        if self.construction_plan.currentText() == 'Castle':
            pixmap_plan = QPixmap('Castle_zugeschnitten.jpg')
            self.plan = "Castle"
            self.label_plan.setPixmap(pixmap_plan)
        if self.construction_plan.currentText() == 'Bridge': 
            pixmap_plan = QPixmap('Bruecke_zugeschnitten.jpg')
            self.label_plan.setPixmap(pixmap_plan)
            self.plan = "Bridge"
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
            self.p.start("python3", ['take_image_btn.py']) #selects file to be executed
        self.pixmap_tl = QPixmap('Images_Results/image.jpg')
        self.label_tl.setPixmap(self.pixmap_tl.scaled(1500, 1500, Qt.KeepAspectRatio))
        self.label_tl.setGeometry(50, 100, 1500, 1500)
        self.label_bl.clear()
        self.label_tr.clear()
        self.label_br.clear()
        
    def process_finished(self):
        self.message("Process finished.")
        self.p = None

    # def show_progress(self, event):
    #     #progress = request_progress()
    #     # assembly_step = open('Images_Results/result.txt','r')
        
    #     # self.text.setText(assembly_step.readline())
    #     # print(assembly_step.readline())
    #     # assembly_step.close()
    #     #self.text.setText('The last completed assembly step is: \n' + self.plan + self.assembly_progress)
    #     if self.new_image_path != " ":
    #         self.change_image()


    # def change_image(self, *args):
    #     # shows the captured image
    #     pixmap = QPixmap(self.folder +"/"+ self.new_image_path + ".jpg")
    #     self.label.setPixmap(pixmap) 
        

    def show_instructions(self):
        info = QMessageBox()
        info.setWindowTitle("Instruction")
        instruction = open('instruction.txt','r')
        
        info.setText("This is the main text!" + "\n" + instruction.read())
        instruction.close()
        x = info.exec_()  # this will show a messagebox 

    # def request_progress(self):
    #     self. assembly_progress = importierte_methode()

    # def define_progress(self):
    #     picture = cv2.imread("/home/sophia/workspace/FuE/Fotos/ids_pyramide/pyramide2.jpg", cv2.IMREAD_UNCHANGED)
    #     picture = extract_plate(picture)
    #     circles = detect_circles(picture, True)
    #     height, width, channel = circles[1].shape
    #     bytesPerLine = 3 * width
    #     q_circles = QImage(circles[1].data, width, height, bytesPerLine, QImage.Format_RGB888)
    #     self.pixmap_tr = QPixmap(q_circles)
    #     self.label_tr.setPixmap(self.pixmap_tr.scaled(700, 700, Qt.KeepAspectRatio))
    #     # assembly_step = open('assembly_step.txt','r')
        
    #     # self.text.setText('Test' + assembly_step.readline())

    #     # assembly_step.close()
    #     with open ('result.txt', 'r') as f:
    #         line = f.read()
    #     self.text.setText(line)
        

        #template_matrix_list, template_name_list = open_saved_matrix()
        #matching_template = detect_matching_template(picture, circles[0], template_matrix_list, template_name_list)

    @pyqtSlot()    
    def detect_progress(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            self.p.start("python3", ['detect_btn.py']) #selects file to be executed
        with open ('Images_Results/result.txt', 'r') as f:
            line = f.read()
        self.text.setText(line)
        self.pixmap_tr = QPixmap('Images_Results/circles.jpg')
        self.label_tr.setPixmap(self.pixmap_tr.scaled(700, 700, Qt.KeepAspectRatio))
        self.pixmap_tl = QPixmap('Images_Results/image.jpg')
        self.label_tl.setPixmap(self.pixmap_tl.scaled(700, 700, Qt.KeepAspectRatio))
        self.pixmap_br = QPixmap('Images_Results/result.jpg')
        self.label_br.setPixmap(self.pixmap_br.scaled(700, 700, Qt.KeepAspectRatio))
        self.pixmap_bl = QPixmap('Images_Results/color_matrix.png')
        self.label_bl.setPixmap(self.pixmap_bl.scaled(700, 700, Qt.KeepAspectRatio))
 
if __name__ == '__main__':        

    app = QApplication(sys.argv)
    w = Fenster()

    sys.exit(app.exec_())
