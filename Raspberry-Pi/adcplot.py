import matplotlib.pyplot as plt

f = open('../adc3.txt', 'r')
adc = []
for line in f.readlines():
    adc.append(int(line))
plt.plot(adc)
plt.show()