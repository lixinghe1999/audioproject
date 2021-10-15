import matplotlib.pyplot as plt

f = open('adc.txt', 'r')
adc = []
for line in f.readlines():
    try:
        adc.append(int(line))
    except:
        pass
plt.plot(adc)
plt.show()