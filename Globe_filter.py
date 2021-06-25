import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data0 = pd.read_csv("D:/Globe signals/305.csv")
print(data0)
data = pd.read_csv("D:/Globe signals/301.csv").iloc[:, 0]
# print(data)
plt.plot(data[:100])
plt.show()
print(len(data)/187)

dow = data[:186]
c = np.fft.rfft(dow)
N = len(c) #number of elements
print('Number of Fourier Coefficients =', N)
# print(c)
plt.plot(c)
plt.show()

#Case 1
# k = 50
c[int(round(N*0.25+1)):] = 0 #set all but not first 10% of elements to zero
y1 = np.fft.irfft(c) #inverse fourier transform of the 10%

#Case 2
c_zero = np.zeros(N, complex) #zero array of fourier coeff.
y2 = np.fft.irfft(c_zero) #inverse fourier transform of the null array
#construct plot
# plt.figure(figsize = [10,5])
plt.plot(dow, 'r', label = 'Original Data') #original data
plt.legend()
plt.show()
plt.plot(y1, 'b' ,label = '10% Fourier Coefficient') #case 1

#display plot
plt.legend()
plt.show()
