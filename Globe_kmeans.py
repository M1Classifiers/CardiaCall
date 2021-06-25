from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load Data
data = pd.read_csv("D:/signal_partitioned_concat.csv")


def model_builder(n_clusters):
    pca = PCA(2)

    # Transform the data
    df = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    label = kmeans.predict(df)

    # plt.scatter(range(len(label)), label)
    # plt.show()

    # filter rows of original data
    filtered_label0 = df[label == 0]
    filtered_label1 = df[label == 1]
    filtered_label2 = df[label == 2]
    filtered_label3 = df[label == 3]

    # Plotting the results
    plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], color='red', label="normal")
    plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color='black')
    plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='black')
    plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1], color='black', label="abnormal")
    plt.legend()
    plt.xlabel("Principal Component X")
    plt.ylabel("Principal Component Y")
    plt.show()
    return label


prediction = model_builder(4)
signal = data.T


def trial_plotter():
    N = 7000
    M = 500
    label = str(N) + ", " + str(prediction[N])
    plt.plot(signal[N].values[1:], label=label)
    plt.legend()
    plt.show()
    label_M = str(M) + ", " + str(prediction[M])
    plt.plot(signal[M].values[1:], label=label_M)
    plt.legend()
    plt.show()


N = 1000
# print(len(data))
while N < len(data):
    label = str(N) + ", " + str(prediction[N])
    plt.plot(signal[N].values[1:], label=label)
    plt.legend()
    plt.show()
    N += 2000

