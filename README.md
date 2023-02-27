# F-E_Projekt_Montage
Projekt Repo for automatic detection of montage steps for simple lego structures


# Installation IDS Software to take a picture with IDS camera

1. Download IDS Software Suite 4.96.1 from this site:
  https://de.ids-imaging.com/download-details/AB02491.html?os=linux&version=&bus=64&floatcalc=
  
2. Start the installation using the archive tar according to this manual:
  https://de.ids-imaging.com/files/downloads/ids-software-suite/readme/readme-ids-software-suite-linux-4.96.1_EN.html
  
  
3. Scroll to Section "Using the "run" script" and proceede the steps.
   You probably need to start the ueyesetup as sudo. Make sure, that all dependencies on the manual in Step 2 are fullfilled.
   Check vie "dpkg --list" if you have the packages "ueye-api" and "ueye-common" installed.
  
4. pip install pyueye

5. Test the camera with the now installe IDS Cameramanager or e.g. with the example script "SimpleLive_PyuEye_OpenCV.py"
