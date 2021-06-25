import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def signal_file_finder(file):
    data = pd.read_csv("D:/Globe signals/" + str(file) + ".csv").iloc[:, 0]

    def fourier_filter(dow):
        c = np.fft.rfft(dow)
        N = len(c)

        c[int(round(N * 0.25 + 1)):] = 0
        y1 = np.fft.irfft(c)

        return y1

    signal = fourier_filter(data)
    i = 0
    mul = 1
    part = 1000
    signal_partitioned = []
    while mul * part < len(signal):
        signal_part = []
        while i < mul * part:
            signal_part += [signal[i]]
            i += 1
        signal_partitioned += [signal_part]
        mul += 1
    signal_partitioned = pd.DataFrame(signal_partitioned)
    return signal_partitioned


signal_files_to_be_concat = []
file = 300
while file < 315:
    signal_file = signal_file_finder(file)
    signal_files_to_be_concat += [signal_file]
    file += 1
signal_partitioned_concat = pd.concat(signal_files_to_be_concat)
signal_partitioned_concat.to_csv("D:/signal_partitioned_concat.csv")




