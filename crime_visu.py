import pandas as pd
import numpy
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#from pylab import *
import matplotlib.pyplot as plt



def cleanData(df):
    print("removing lines with incorrect gps values...")
    return df[df['Y'] < 60]

if __name__ == "__main__":
    loc_train = "./data/train.csv"

    print("reading train data")
    df_train = pd.read_csv(loc_train)
    #print(df_train)
    df_train = cleanData(df_train)
    #df_train = df_train[df_train['Y'] < 60]

    #df_train[df_train['X'] >= 60]
    print(df_train)
    plt.scatter(df_train.X, df_train.Y)
    plt.show()
    # plot(df_train["X"],df_train["C"])
    # df_train.
