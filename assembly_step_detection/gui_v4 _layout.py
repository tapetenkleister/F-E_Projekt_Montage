import os
import cv2
import sys
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from functions import *


class Fenster(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initMe()
        
    def initMe(self):
        self.p = None
        self.text = QLabel(' ', self)
        self.text.setGeometry(0,1600, 1000, 50)
        hight = 700
        width = 700
    
        # creates label for the captured image
        self.label_tl = QLabel(self)
        self.label_tl.setText("image")
        self.label_tl.setFont(QFont('Arial', 16))   
        
        # creates label for the cropped image with circles
        self.label_tr = QLabel(self)
        self.label_tr.setText("cropped image with circles")
        self.label_tr.setFont(QFont('Arial', 16))      
        
        #creates label for the colored matrix
        self.label_bl = QLabel(self)
        self.label_bl.setText("colored matrix")
        self.label_bl.setFont(QFont('Arial', 16))
        
        #creates label for the cropped and highlighted image
        self.label_br = QLabel(self)
        self.label_br.setText("cropped and highlighted image")
        self.label_br.setFont(QFont('Arial', 16))
 
        # shows an image of the selected construction plan
        self.label_plan = QLabel(self)
        self.pixmap_plan = QPixmap('Templates/Pyramid/Pyramid_thumbnail.png')
        self.label_plan.setPixmap(self.pixmap_plan.scaled(700,700,Qt.KeepAspectRatio))

        # creates an selection field for the construction plan
        self.construction_plan = QComboBox(self)
        self.construction_plan.setFixedSize(QSize(300, 100))
        self.construction_plan.setToolTip('Shows the available templates')
        # call addItem() for each folder in /Templates
        for folder in os.listdir('Templates'):
            self.construction_plan.addItem(folder)
        self.construction_plan.currentIndexChanged.connect(self.select_construct_plan)

        # creates a button to to show the current progress step
        button_progress = QPushButton('Show progress', self)
        button_progress.setFixedSize(QSize(300, 100))
        button_progress.setToolTip('Shows the current <b>progress</b> for the selected plan')
        button_progress.clicked.connect(self.detect_progress)

        # creates a button to capture an image
        button_photo = QPushButton('Take photo', self)
        button_photo.setFixedSize(QSize(300, 100))
        button_photo.setToolTip('take <b>Photo</b>')
        button_photo.clicked.connect(self.start_process)
        # self.text2 = QPlainTextEdit()
        # self.text2.setReadOnly(True)
        # self.text2.setGeometry(1700, 500, 200, 200)

        # button to save new construction template
        button_save = QPushButton('Save new template', self)
        button_save.setToolTip('save <b>new template</b>')
        button_save.setFixedSize(QSize(300, 100))
        button_save.clicked.connect(self.save_new_plan)

        # creating action for closing the window
        exitMe = QAction('&Exit', self)
        exitMe.setShortcut('Ctrl+E')
        exitMe.triggered.connect(self.close)

        # creating action for capturing an image
        capture_picture = QAction('take &photo', self)
        capture_picture.setShortcut('Ctrl+F')
        capture_picture.triggered.connect(self.start_process)

        # creating action for showing information
        instruction = QAction('&Instructions', self)
        instruction.setShortcut('Ctrl+I')
        instruction.triggered.connect(self.show_instructions)

        # # creating action for changing save folder
        # change_folder = QAction('change &saving location',self)
        # change_folder.triggered.connect(self.change_folder)
        # change_folder.setShortcut('Ctrl+S')

        # creates a menu bar
        menuBar = self.menuBar()
        menu = menuBar.addMenu('&Menu')
 
        menu.addAction(capture_picture)
        #menu.addAction(show_progress)
        #menu.addAction(change_folder)
        menu.addAction(instruction)
        menu.addAction(exitMe)

        self.statusBar().showMessage('Assembly inspection from Stephan Klotz, Jennifer Sissi Lange, Sophia Reiter')
        widget = QWidget()
        # Create an outer layout
        outerLayout = QHBoxLayout(widget)
        # Create a layout to show the progress
        leftLayout = QVBoxLayout(widget)
        # Create a layout for the upper images
        topLeftLayout = QHBoxLayout()
        # Create a layout for the lower images
        lowLeftLayout = QHBoxLayout()
        # Create a layout for buttons
        rightLayout = QVBoxLayout(widget)
        # Set the alignment of the right layout
        rightLayout.setAlignment(Qt.AlignHCenter)

        # Add labels the leftLayout
        topLeftLayout.addWidget(self.label_tl)
        topLeftLayout.addWidget(self.label_tr)
        lowLeftLayout.addWidget(self.label_bl)
        lowLeftLayout.addWidget(self.label_br)
        leftLayout.addLayout(topLeftLayout)
        leftLayout.addLayout(lowLeftLayout)
        # add buttopns, combobox and label to the rightLayout
        rightLayout.addWidget(self.label_plan)
        rightLayout.addWidget(self.construction_plan)
        rightLayout.addWidget(button_photo)
        rightLayout.addWidget(button_progress)
        rightLayout.addWidget(button_save)
        rightLayout.addWidget(self.text)
       
        outerLayout.addLayout(leftLayout)
        outerLayout.addLayout(rightLayout)
        self.setCentralWidget(widget)

        # Set the window's main layout
        self.setLayout(outerLayout)
        self.setGeometry(100, 100, 2750, 2000)
        self.setWindowTitle('Assembly inspection')
        self.show()
      

    # def change_folder(self):
    #     # open the dialog to select path
    #     path = QFileDialog.getExistingDirectory(self, 
    #                                             "Picture Location", "")
    #     # if path is selected
    #     if path:
    #         # update the path
    #         self.folder = path

    def select_construct_plan(self, i):
        # set thumbnail.png matching the construction_plan.currentText() as pixmap
        for folder in os.listdir('Templates'):
            print('1' + folder)
            print('2' + self.construction_plan.currentText())
            if self.construction_plan.currentText() == folder:
                pixmap_plan = QPixmap('Templates/' + self.construction_plan.currentText() + '/' + self.construction_plan.currentText() + '_thumbnail.png')
                #self.label_plan.setPixmap(pixmap_plan)
                self.label_plan.setPixmap(pixmap_plan.scaled(700,700,Qt.KeepAspectRatio))
        #print(self.construction_plan.currentText())

    #def message(self, s):
    #    self.text2.appendPlainText(s)

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
        #self.message("Process finished.")
        self.p = None


    def save_new_plan(self):
        # user should be able to enter the name of the new template and the amount of nubs on the longest side
        text, ok = QInputDialog.getText(self, 'save new plan', 'Please enter template name (with a capital letter):') # entering the file name
        self.template_name = text
        
        if ok:
            number, ok = QInputDialog.getInt(self, 'save new plan', 'Please enter the amount of nubs on the longest side: ') # entering the file name
            self.longest_side = number
            
            if ok: 
                newpath = 'Templates/' + self.template_name
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                # open the dialog to select path
                path = QFileDialog.getExistingDirectory(self, 
                                                        "Picture Location", "")
                # if path is selected
                if path:
                    # update the path
                    self.folder = path

                #select folder and copy images in selected folder
                for file in os.listdir(self.folder):
                    if file.endswith(".png"):
                        shutil.copy(self.folder +"/"+ file, newpath)
                # crreate json file for the new template
                new_matrix = safe_new_matrix(self.template_name, self.longest_side)
            
        else:
            # show gui again
            self.show()
                

    def show_instructions(self):
        # show instructions how to use the application
        info = QMessageBox()
        info.setWindowTitle("Instruction")
        instruction = open('Docs/instructions.txt','r')
        info.setText("This is the main text!" + "\n" + instruction.read())
        instruction.close()
        x = info.exec_()


    @pyqtSlot()    
    def detect_progress(self):
        if self.p is None:  # No process running.
            #self.message("Executing process")
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
