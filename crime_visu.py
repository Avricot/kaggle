import pandas as pd
import numpy
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#from pylab import *
import matplotlib.pyplot as plt





if __name__ == "__main__":
    loc_train = "./data/train.csv"

    print("reading train data")
    df_train = pd.read_csv(loc_train)
    print(df_train["X"])
    df_train.plot(x='X', y='Y', style='o')
    plt.show()
    # plot(df_train["X"],df_train["C"])
    # df_train.
