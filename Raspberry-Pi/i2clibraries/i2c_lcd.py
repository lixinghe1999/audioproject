from i2clibraries import i2c
from time import *

class i2c_lcd:

	# Commands
	CMD_Clear_Display = 0x01
	CMD_Return_Home = 0x02
	CMD_Entry_Mode = 0x04
	CMD_Display_Control = 0x08
	CMD_Cursor_Display_Shift = 0x10
	CMD_Function_Set = 0x20
	CMD_DDRAM_Set = 0x80

	# Options
	OPT_Increment = 0x02 					# CMD_Entry_Mode
	OPT_Display_Shift = 0x01 				# CMD_Entry_Mode
	OPT_Enable_Display = 0x04 				# CMD_Display_Control
	OPT_Enable_Cursor = 0x02 				# CMD_Display_Control
	OPT_Enable_Blink = 0x01 				# CMD_Display_Control
	OPT_Display_Shift = 0x08 				# CMD_Cursor_Display_Shift
	OPT_Shift_Right = 0x04 					# CMD_Cursor_Display_Shift 0 = Left
	OPT_2_Lines = 0x08 						# CMD_Function_Set 0 = 1 line
	OPT_5x10_Dots = 0x04 					# CMD_Function_Set 0 = 5x7 dots
	

	def __init__(self, addr, port, en, rw, rs, d4, d5, d6, d7, backlight = -1):
		self.bus = i2c.i2c(port, addr)

		self.en = en
		self.rs = rs
		self.rw = rw
		self.d4 = d4
		self.d5 = d5
		self.d6 = d6
		self.d7 = d7
		self.backlight = backlight

		self.backlight_state = False

		# Activate LCD
		initialize_i2c_data = 0x00
		initialize_i2c_data = self._pinInterpret(self.d4, initialize_i2c_data, 0b1)
		initialize_i2c_data = self._pinInterpret(self.d5, initialize_i2c_data, 0b1)
		self._enable(initialize_i2c_data)	
		sleep(0.2)
		self._enable(initialize_i2c_data)
		sleep(0.1)
		self._enable(initialize_i2c_data)
		sleep(0.1)	

		# Initialize 4-bit mode
		initialize_i2c_data = self._pinInterpret(self.d4, initialize_i2c_data, 0b0)
		self._enable(initialize_i2c_data)
		sleep(0.01)

		self.command(self.CMD_Function_Set | self.OPT_2_Lines)
		self.command(self.CMD_Display_Control | self.OPT_Enable_Display | self.OPT_Enable_Cursor)
		self.command(self.CMD_Clear_Display)
		self.command(self.CMD_Entry_Mode | self.OPT_Increment |  self.OPT_Display_Shift) 
	
	def clear(self):
		self.command(self.CMD_Clear_Display)
		sleep(0.1)
	
	def home(self):
		self.command(self.CMD_Return_Home)
		sleep(0.1)

	def setPosition(self, line, pos):
		if line == 1:
			address = pos
		elif line == 2:
			address = 0x40 + pos
		elif line == 3:
			address = 0x14 + pos
		elif line == 4:
			address = 0x54 + pos
		self.command(self.CMD_DDRAM_Set + address)
	
	def writeChar(self, char):
		self._write(ord(char), False)

	def writeString(self, string):
		for c in string:
			self.writeChar(c)
      
	def backLightOn(self):
		if self.backlight >= 0:
			self.bus.write_byte(self._pinInterpret(self.backlight, 0x00, 0b1))
			self.backlight_state = True

	def backLightOff(self):
		if self.backlight >= 0:	
			self.bus.write_byte(self._pinInterpret(self.backlight, 0x00, 0b0))	
			self.backlight_state = False

	def _write(self, data, command=True):
		i2c_data = 0x00

		#Add data for high nibble
		hi_nibble = data >> 4
		i2c_data = self._pinInterpret(self.d4, i2c_data, (hi_nibble & 0x01))
		i2c_data = self._pinInterpret(self.d5, i2c_data, ((hi_nibble >> 1) & 0x01))
		i2c_data = self._pinInterpret(self.d6, i2c_data, ((hi_nibble >> 2) & 0x01))
		i2c_data = self._pinInterpret(self.d7, i2c_data, ((hi_nibble >> 3) & 0x01))

		# Set the register selector to 1 if this is data
		if command != True:
			i2c_data = self._pinInterpret(self.rs, i2c_data, 0x1)

		# Toggle Enable 
		self._enable(i2c_data)

		i2c_data = 0x00

		#Add data for high nibble
		low_nibble = data & 0x0F 
		i2c_data = self._pinInterpret(self.d4, i2c_data, (low_nibble & 0x01))
		i2c_data = self._pinInterpret(self.d5, i2c_data, ((low_nibble >> 1) & 0x01))
		i2c_data = self._pinInterpret(self.d6, i2c_data, ((low_nibble >> 2) & 0x01))
		i2c_data = self._pinInterpret(self.d7, i2c_data, ((low_nibble >> 3) & 0x01))

		# Set the register selector to 1 if this is data
		if command != True:
			i2c_data = self._pinInterpret(self.rs, i2c_data, 0x1)
		
		self._enable(i2c_data)
		
		sleep(0.01)
		
	def _pinInterpret(self, pin, data, value="0b0"):
		if value:
			# Construct mask using pin
			mask = 0x01 << (pin)  
			data = data | mask
		else:
			# Construct mask using pin
			mask = 0x01 << (pin) ^ 0xFF
			data = data & mask
		return data 
	
	def _enable(self, data):
		# Determine if black light is on and insure it does not turn off or on
		if self.backlight_state:
			data = self._pinInterpret(self.backlight, data, 0b1)
		else:
			data = self._pinInterpret(self.backlight, data, 0b0)
		
		self.bus.write_byte(data)
		self.bus.write_byte(self._pinInterpret(self.en, data, 0b1))
		self.bus.write_byte(data)
	
	# For legacy
	def command(self, data):
		self._write(data)

