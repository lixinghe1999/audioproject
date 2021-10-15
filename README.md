# audioproject
## sensors involve: GY85 BMI160 piezo disc piezo vibration sensor
## platform: raspberry pi 4b
## structure of the project
```
├─ README.md
├─ Raspberry-Pi
│    ├─ BMI160_i2c  ## for BMI160 I2C control
│    │    ├─ __init__.py
│    │    ├─ commands.py
│    │    ├─ definitions.py
│    │    ├─ registers.py
│    │    └─ sleep.py
│    ├─ adc        ## for MCP3008 ADC control
│    │    └─ MCP3008.py
│    ├─ bmi160.py
│    ├─ datarecord.py
│    ├─ gy85.py
│    ├─ i2clibraries  ## libraries for GY-85 
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
├─ airpods ## airpods collect data
├─ bmiacc_1.txt
├─ bmiacc_2.txt
├─ compass.txt
├─ denoiser
├─ exp1  ## exp1 - use GY-85
│    ├─ HE
│    │    ├─ 55db
│    │    ├─ 70db
│    │    └─ mask
│    └─ HOU
│           ├─ 55db
│           ├─ 70db
│           └─ mask
├─ exp2  ## exp2 - use BMI-160
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
├─ processing.py ## main function
├─ project proposal.pptx ## only slide
├─ quick2wire-python-api ## i2c libraries
├─ reference ## generate noise data
