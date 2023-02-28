# F-E_Projekt_Montage
Projekt Repo for automatic detection of montage steps for simple lego structures


# Installation IDS Software to take a picture with IDS camera

1. Download IDS Software Suite 4.96.1 für Linux 64-Bit - Archivdatei from this site (a logged in account is necessary):
    https://de.ids-imaging.com/download-details/AB02491.html?os=linux&version=&bus=64&floatcalc=
    
2. Extract the files in a folder of you choice.
  
  
3. Start the installation using the files according to this manual:
  https://de.ids-imaging.com/files/downloads/ids-software-suite/readme/readme-ids-software-suite-linux-4.96.1_EN.html
  
  
4. Scroll to Section "Using the "run" script" and proceede the steps.
   You probably need to start the ueyesetup as sudo. Make sure, that all dependencies on the manual in Step 3 are fullfilled.
  
5. pip install pyueye

6. Test the camera with the now installed IDS Cameramanager or e.g. with the example script "SimpleLive_PyuEye_OpenCV.py"
