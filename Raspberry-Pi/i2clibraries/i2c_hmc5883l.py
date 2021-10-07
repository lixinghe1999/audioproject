import math
from i2clibraries import i2c
from time import *

class i2c_hmc5883l:
    
    ConfigurationRegisterA = 0x00
    ConfigurationRegisterB = 0x01
    ModeRegister = 0x02
    AxisXDataRegisterMSB = 0x03
    AxisXDataRegisterLSB = 0x04
    AxisZDataRegisterMSB = 0x05
    AxisZDataRegisterLSB = 0x06
    AxisYDataRegisterMSB = 0x07
    AxisYDataRegisterLSB = 0x08
    StatusRegister = 0x09
    IdentificationRegisterA = 0x10
    IdentificationRegisterB = 0x11
    IdentificationRegisterC = 0x12
    

    MeasurementContinuous = 0x00
    MeasurementSingleShot = 0x01
    MeasurementIdle = 0x03
    
    def __init__(self, port, addr=0x1e, gauss=1.3):
        self.bus = i2c.i2c(port, addr)
        
        self.setScale(gauss)
        
    def __str__(self):
        ret_str = ""
        (x, y, z) = self.getAxes()
        ret_str += "Axis X: "+str(x)+"\n"       
        ret_str += "Axis Y: "+str(y)+"\n" 
        ret_str += "Axis Z: "+str(z)+"\n" 
        
        ret_str += "Declination: "+self.getDeclinationString()+"\n" 
        
        ret_str += "Heading: "+self.getHeadingString()+"\n" 
        
        return ret_str
        
        
        
    def setContinuousMode(self):
        self.setOption(self.ModeRegister, self.MeasurementContinuous)
        
    def setScale(self, gauss):
        if gauss == 0.88:
            self.scale_reg = 0x00
            self.scale = 0.73
        elif gauss == 1.3:
            self.scale_reg = 0x01
            self.scale = 0.92
        elif gauss == 1.9:
            self.scale_reg = 0x02
            self.scale = 1.22
        elif gauss == 2.5:
            self.scale_reg = 0x03
            self.scale = 1.52
        elif gauss == 4.0:
            self.scale_reg = 0x04
            self.scale = 2.27
        elif gauss == 4.7:
            self.scale_reg = 0x05
            self.scale = 2.56
        elif gauss == 5.6:
            self.scale_reg = 0x06
            self.scale = 3.03
        elif gauss == 8.1:
            self.scale_reg = 0x07
            self.scale = 4.35
        
        self.scale_reg = self.scale_reg << 5
        self.setOption(self.ConfigurationRegisterB, self.scale_reg)
        
    def setDeclination(self, degree, min = 0):
        self.declinationDeg = degree
        self.declinationMin = min
        self.declination = (degree+min/60) * (math.pi/180)
        
    def setOption(self, register, *function_set):
        options = 0x00
        for function in function_set:
            options = options | function
        self.bus.write_byte(register, options)
        
    # Adds to existing options of register  
    def addOption(self, register, *function_set):
        options = self.bus.read_byte(register)
        for function in function_set:
            options = options | function
        self.bus.write_byte(register, options)
        
    # Removes options of register   
    def removeOption(self, register, *function_set):
        options = self.bus.read_byte(register)
        for function in function_set:
            options = options & (function ^ 0b11111111)
        self.bus.write_byte(register, options)
        
    def getDeclination(self):
        return (self.declinationDeg, self.declinationMin)
    
    def getDeclinationString(self):
        return str(self.declinationDeg)+"\u00b0 "+str(self.declinationMin)+"'"
    
    # Returns heading in degrees and minutes
    def getHeading(self):
        (scaled_x, scaled_y, scaled_z) = self.getAxes()
        
        headingRad = math.atan2(scaled_y, scaled_x)
        headingRad += self.declination

        # Correct for reversed heading
        if(headingRad < 0):
            headingRad += 2*math.pi
            
        # Check for wrap and compensate
        if(headingRad > 2*math.pi):
            headingRad -= 2*math.pi
            
        # Convert to degrees from radians
        headingDeg = headingRad * 180/math.pi
        degrees = math.floor(headingDeg)
        minutes = round(((headingDeg - degrees) * 60))
        return (degrees, minutes)
    
    def getHeadingString(self):
        (degrees, minutes) = self.getHeading()
        return str(degrees)+"\u00b0 "+str(minutes)+"'"
        
    def getAxes(self):
        (magno_x, magno_z, magno_y) = self.bus.read_3s16int(self.AxisXDataRegisterMSB)

        if (magno_x == -4096):
            magno_x = None
        else:
            magno_x = round(magno_x * self.scale, 4)
            
        if (magno_y == -4096):
            magno_y = None
        else:
            magno_y = round(magno_y * self.scale, 4)
            
        if (magno_z == -4096):
            magno_z = None
        else:
            magno_z = round(magno_z * self.scale, 4)
            
        return (magno_x, magno_y, magno_z)
        
    
        
    
