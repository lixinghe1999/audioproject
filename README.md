# audioproject
## sensors involve: GY85 BMI160 piezo disc piezo vibration sensor
## platform: raspberry pi 4b
## structure of the project
├─ README.md
├─ Raspberry-Pi
│    ├─ BMI160_i2c
│    │    ├─ __init__.py
│    │    ├─ commands.py
│    │    ├─ definitions.py
│    │    ├─ registers.py
│    │    └─ sleep.py
│    ├─ adc
│    │    └─ MCP3008.py
│    ├─ bmi160.py
│    ├─ datarecord.py
│    ├─ gy85.py
│    ├─ i2clibraries
│    │    ├─ __init__.py
│    │    ├─ __pycache__
│    │    ├─ i2c.py
│    │    ├─ i2c.pyc
│    │    ├─ i2c_adxl345.py
│    │    ├─ i2c_hmc5883l.py
│    │    ├─ i2c_itg3205.py
│    │    ├─ i2c_itg3205.pyc
│    │    ├─ i2c_l3g4200.py
│    │    ├─ i2c_lcd.py
│    │    └─ i2c_lcd_smbus.py
│    ├─ mic.py
│    └─ vibration.wav
├─ acc.txt
├─ adc.txt
├─ adcplot.py
├─ addnoise.py
├─ airpods
├─ bmiacc_1.txt
├─ bmiacc_2.txt
├─ compass.txt
├─ denoiser
├─ exp1
│    ├─ HE
│    │    ├─ 55db
│    │    ├─ 70db
│    │    └─ mask
│    └─ HOU
│           ├─ 55db
│           ├─ 70db
│           └─ mask
├─ exp2
|    ├─ HE
│    └─ HOU      
├─ figure
│    ├─ 55db.png
│    ├─ 70db.png
│    ├─ calibrate.png
│    ├─ fail.png
│    ├─ no_calibrate.png
│    ├─ phase_acc.png
│    └─ phase_random.png
├─ imuplot.py
├─ micplot.py
├─ microphone.py
├─ processing.py
├─ project proposal.pptx
├─ quick2wire-python-api
├─ reference
