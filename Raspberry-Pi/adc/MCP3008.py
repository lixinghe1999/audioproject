import spidev
import time
class MCP3008:
    def __init__(self, bus=0, device=0):
        self.bus, self.device = bus, device
        self.spi = spidev.SpiDev()
        self.open()
        self.spi.max_speed_hz = 1000000  # 1MHz

    def open(self):
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = 1000000  # 1MHz

    def read(self, channel=0):
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data

    def close(self):
        self.spi.close()
if __name__ == "__main__":
    adc = MCP3008()
    f = open('adc.txt', 'w')
    while (1):
        f.write(str(adc.read(channel = 0 )) + '\n')
        #print("Applied voltage: %.2f" % (adc.read( channel = 0 ) / 1023.0 * 3.3) )