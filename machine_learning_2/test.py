import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import sklearn as sk

from urllib.request import urlopen

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

def check_versions():
    # Check the versions of libraries
    print("pandas version: {}".format(pd.__version__))
    print("matplotlib version: {}".format(matplotlib.__version__))
    print("numpy version: {}".format(np.__version__))
    print("scipy version: {}".format(sp.__version__))
    print("sklearn version: {}".format(sk.__version__))

check_versions()


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
def read_csv_with_pandas(url):
    # Load dataset
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    print("Read with Pandas", dataset.shape)
    return dataset

def read_csv_with_numpy(url):
    # Load dataset
    dataset = np.genfromtxt(url, delimiter=",", dtype="U75")
    print("With Numpy", dataset.shape)
    return dataset


def read_csv_native(url):
    # Load dataset
    raw_data = urlopen(url)
    data = [line.strip().decode('utf-8').split(",") for line in raw_data if line.strip()]
    dataset = [list(row) for row in data]
    print("Native with Python", len(dataset), len(dataset[0]))
    return dataset

read_csv_with_pandas(url)
read_csv_with_numpy(url)
read_csv_native(url)