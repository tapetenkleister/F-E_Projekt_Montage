import os
import cv2
import sys
import shutil
import time
import subprocess
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
        self.result = QLabel(' ', self)
        self.result.setGeometry(0,1600, 1000, 50)      
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

        # button to to show the current progress step
        button_progress = QPushButton('Detect assembly step', self)
        button_progress.setFixedSize(QSize(300, 100))
        button_progress.setToolTip('Shows the current <b>progress</b> for the selected plan')
        button_progress.clicked.connect(self.detect_progress)

        # button to capture an image
        button_photo = QPushButton('Take photo', self)
        button_photo.setFixedSize(QSize(300, 100))
        button_photo.setToolTip('take <b>Photo</b>')
        button_photo.clicked.connect(self.start_process)

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

        # creates a menu bar
        menuBar = self.menuBar()
        menu = menuBar.addMenu('&Menu')
 
        menu.addAction(capture_picture)
        menu.addAction(instruction)
        menu.addAction(exitMe)

        self.statusBar().showMessage('Assembly inspection from Stephan Klotz, Jennifer Sissi Lange, Sophia Reiter')

        # Create a widget for the window's main layout
        widget = QWidget()
        # Create an outer layout
        outerLayout = QHBoxLayout(widget)
        # Create a layout to show the assembly step
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
        # Add buttons, combobox and label to the rightLayout
        rightLayout.addWidget(self.label_plan)
        rightLayout.addWidget(self.construction_plan)
        rightLayout.addWidget(button_photo)
        rightLayout.addWidget(button_progress)
        rightLayout.addWidget(button_save)
        rightLayout.addWidget(self.result)
        # Add the layouts to the outer layout
        outerLayout.addLayout(leftLayout)
        outerLayout.addLayout(rightLayout)
        self.setCentralWidget(widget)
        # Set the window's main layout
        self.setLayout(outerLayout)
        self.setGeometry(100, 100, 2750, 2000)
        self.setWindowTitle('Assembly inspection')
        self.show()


    def select_construct_plan(self, i):
        """ This function is called when the user selects a construction plan from the combobox.
        It shows the corresponding image (thumbnail) of the selected plan.
        Args:
            i: index of the selected item of the combobox
        """     
        for folder in os.listdir('Templates'):
            if self.construction_plan.currentText() == folder:
                pixmap_plan = QPixmap('Templates/' + self.construction_plan.currentText() + '/' + self.construction_plan.currentText() + '_thumbnail.png')
                self.label_plan.setPixmap(pixmap_plan.scaled(700,700,Qt.KeepAspectRatio))


    @pyqtSlot()    
    def start_process(self):
        """ This function is called when the user clicks on the button to capture an image.
        It starts the process to capture an image and shows the image in the corresponding label.
        """     
        # if no process is running
        if self.p is None:  
            # self.message("Executing process")
            # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p = QProcess()  
            # Clean up once complete
            self.p.finished.connect(self.process_finished)  
            #selects file to be executed
            self.p.start("python3", ['take_image_btn.py']) 
            # waits until file is executed
            self.p.waitForFinished()
            # shows the results
            self.pixmap_tl = QPixmap('Images_Results/image.jpg')
            self.label_tl.setPixmap(self.pixmap_tl.scaled(1500, 1500, Qt.KeepAspectRatio))
            self.label_tl.setGeometry(50, 100, 1500, 1500)
            self.label_bl.clear()
            self.label_tr.clear()
            self.label_br.clear()

        
    def process_finished(self):
        """ This function is called when the process to capture an image or to detect assembly step is finished.
        """
        self.p = None


    def save_new_plan(self):
        """ This function creates a new folder in the Templates folder and copies all images from the selected folder into the new folder.
        It calls the function to save a new matrix. 
        """       
        # input dialog to enter the name of a new template 
        text, ok = QInputDialog.getText(self, 'save new plan', 'Please enter template name (with a capital letter):') # entering the file name
        self.template_name = text
        # input dialog to enter the amount of nubs on the longest side
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
                # create json file for the new template
                new_matrix = safe_new_matrix(self.template_name, self.longest_side)
        else:
            # show gui again
            self.show()
                

    def show_instructions(self):
        """ This function shows the instructions how to use the application.
        """   
        # create a message box     
        info = QMessageBox()
        info.setWindowTitle("Instruction")
        instruction = open('Docs/instructions.txt','r')
        # read the instructions from the file and show them in the message box
        info.setText("This is the main text!" + "\n" + instruction.read())
        instruction.close()
        x = info.exec_()


    @pyqtSlot()    
    def detect_progress(self):
        """ This function is called when the user clicks on the button to detect the progress.
        It starts the process to detect the progress and shows the result in the corresponding label.
        """
        if self.p is None:  # No process running.
            #self.message("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            self.p.start("python3", ['detect_btn.py']) #selects file to be executed
            # make sure detect_btn.py is executed before the following code is executed
            self.p.waitForFinished()
            # read the result.txt file and show the result in the text box
            with open ('Images_Results/result.txt', 'r') as f:
                line = f.read()
            self.result.setText(line)
            self.pixmap_tr = QPixmap('Images_Results/circles.jpg')
            self.label_tr.setPixmap(self.pixmap_tr.scaled(700, 700, Qt.KeepAspectRatio))
            self.pixmap_tl = QPixmap('Images_Results/image.jpg')
            self.label_tl.setPixmap(self.pixmap_tl.scaled(700, 700, Qt.KeepAspectRatio))
            self.pixmap_br = QPixmap('Images_Results/result.jpg')
            self.label_br.setPixmap(self.pixmap_br.scaled(700, 700, Qt.KeepAspectRatio))
            self.pixmap_bl = QPixmap('Images_Results/color_matrix.png')
            self.label_bl.setPixmap(self.pixmap_bl.scaled(700, 700, Qt.KeepAspectRatio))
            from detect_btn import detected_assembly_step
            detected_template = detected_assembly_step.split(' ')
            # change index of construction_plan to the detected template[0]
            self.construction_plan.setCurrentIndex(self.construction_plan.findText(detected_template[0]))


 
if __name__ == '__main__':        

    app = QApplication(sys.argv)
    w = Fenster()

    sys.exit(app.exec_())
