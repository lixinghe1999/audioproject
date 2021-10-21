# GY-85_Raspberry-Pi
Interfacing GY-85 (IMU module) with Raspberry Pi 3b+


# Step 1: Updating python version

- Open terminal and write:

      $ sudo apt-get install python3-dev libffi-dev libssl-dev -y
      $ wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tar.xz
      $ tar xJf Python-3.6.3.tar.xz
      $ cd Python-3.6.3
      $ ./configure
      $ make
      $ sudo make install
      $ sudo pip3 install --upgrade pip
      $ sudo nano ~/.bashrc
      $ alias python3=python3.6
      $ source ~/.bashrc
- To check if python has successfully upgraded, write the following lines in terminal:

      $ python3
      $ print("fhd-codes")
        
  (if the message is printed on the terminal, you have python3 installed now.)
- Close the terminal

# Step 2: Enabling i2c on Raspberry Pi
- Goto the following directory:      /etc/modules-load.d/
- Open "modules.conf" file and write **i2c-dev** in the end and save the file
- After that, open the following file:  /etc/modprobe.d/raspi-blacklist.conf
- Comment the line (by adding # in the start) that says **blacklist i2c-bcm2708** and save the file

  If the file is empty, leave it as it is
- Open terminal and enter these lines:

      $ sudo apt-get update && sudo apt-get install i2c-tools    
      $ sudo adduser USER i2c       

  (write username in place of USER which is 'pi' by default)
        
- Update the boot file by:

      $ sudo nano /boot/config.txt
  write the following lines in the end:

      dtparam=i2c_arm=on
      dtparam=i2c1=on
  Press **Ctrl+X**. Then press **Y** and hit **Enter** 

# Step 3: Installing quick2wire library
- Install git by writing following lines in terminal:

      $ sudo apt-get install git
      
- Download quick2wire API by:

      $ git clone https://github.com/quick2wire/quick2wire-python-api.git

- Now, we need to enable Python to access this APi. 
  Open **~/.profile** and add the following line in the end:
    
    **export PYTHONPATH=$PYTHONPATH:$HOME/quick2wire-python-api**
    
    replace $PYTHONPATH with the path where python is installed; and $HOME with the path where your downloaded API is stored.
    
  For example, in my case, it is: **export PYTHONPATH=/home/pi/Python-3.6.3:/home/pi/quick2wire-python-api**
    
  NOTE: **~/** is the slang for directory **/home/pi/**. The **.profile** file will be hidden, so make the hidden files visible first.
  
  Save and close the file
  
- Reboot your Raspberry Pi

      $ sudo reboot

# Step 4: Connecting GY-85 and checking the connection

- Connect GY-85 with Raspberry Pi.

  Connect **3.3v, GND, SCL, SDA** pins on GY-85 with the same pins on Raspberry Pi
- Open terminal and write:

      $ sudo i2cdetect -y 1
   
   where **1** is the port number for newer version of Raspberry Pi. If you're using older model of Raspberry Pi, write **0** instead.
   
   A table will appear which will show the address of connected devices. The sensors have following addresses:
   
   Accelerometer(ADXL): **53**
   
   Gyroscope(ITG): **68**
   
   Compass(HMC): **1e**       (if you see some other address for digital compass, note it down. It is explained in later section below)
   
   
# Step 5: Downloading the code files

- Open the terminal and download the code files

      $ git clone https://github.com/fhd-codes/GY-85_Raspberry-Pi.git
      
# Getting the values from Gyro, Acc, and Compass

- Run the following files to get the data:
  
  gyroTest.py     
  
  accTest.py     
  
  compassTest.py
  
  **Note:** If the output from __*compassTest.py*__ is giving 0 values, it means that GY-85 module has QMC5883l instead of HMC5883l. And you might be getting **0d** instead of **1e** as address in Step 4. Special thanks to this [source](https://forum.arduino.cc/index.php?topic=519387.0) from where I figured out the right problem. If this this your case, download the following library:
  
      $ git clone https://github.com/RigacciOrg/py-qmc5883l.git
      
  Open the folder and run **setup.py** one time. Run **qmcTest.py** instead of **compassTest.py**, and you will start seeing the values.
  
  **Kudos!!** The data is being acquired form GY-85 module.
  
 
# Main reference sources:

- Code is collected from these sources, and edited by me to make it compatible for Raspberry pi 3b+
  
  [source1](https://topic.alibabacloud.com/a/raspberry-pi-connects-9-axis-imu-sensor-gy-85-module_8_8_32153608.html)
  
  [source2](https://elinux.org/RPi_ADC_I2C_Python)
  
  [source3](https://topic.alibabacloud.com/a/raspberry-pi-connects-9-axis-imu-sensor-gy-85-module_8_8_32153608.html)
  
  [source4](http://www.knight-of-pi.org/installing-python3-6-on-a-raspberry-pi/)
