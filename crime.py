import pandas as pd
import numpy
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics


def improveDate(df):
    print("improving dates features")
    # dates = pd.DatetimeIndex(df["Dates"])
    # print("getting years...")
    # df['year'] = dates.year
    # print("getting day of year...")
    # df['day_of_year'] = dates.dayofyear
    # print("getting day of week...")
    # df['day_of_week'] = dates.dayofweek
    # print("getting hour of day...")
    # df['hour_of_day'] = dates.hour
    # print("getting month...")
    # df['month'] = dates.month

    df["Dates"] = pd.to_datetime(df["Dates"], coerce=True)
    # df_train['day_of_year'] = df_train["Dates"].apply(lambda df: (df.timetuple().tm_yday, df.weekday(), df.hour))
    print("getting years...")
    df['year'] = df["Dates"].apply(lambda df: df.year)
    print("getting day of year...")
    df['day_of_year'] = df["Dates"].apply(lambda df: df.timetuple().tm_yday)
    print("getting day of weey...")
    df['day_of_week'] = df["Dates"].apply(lambda df: df.weekday())
    print("getting hour of day...")
    df['hour_of_day'] = df["Dates"].apply(lambda df: df.hour)
    print("getting month...")
    df['month'] = df["Dates"].apply(lambda df: df.month)
    # df['day_of_year'] = df["Dates"].values.astype('datetime64[Y]').astype(int) + 1970
    # df['day_of_week'] = df["Dates"].values.astype('datetime64[M]').astype(int) % 12 + 1
    # df['hour_of_day'] = df["Dates"].values - df["Dates"].values.astype('datetime64[M]') + 1
    normalize(df, ['year', 'day_of_year', 'day_of_week', 'hour_of_day', 'month'])


def improveAddress(df):
    print("improving address features")
    pdDistrictLabel = preprocessing.LabelEncoder()
    pdDistrictLabel.fit(df["PdDistrict"])
    df["PdDistrict"] = pdDistrictLabel.transform(df["PdDistrict"]).astype(numpy.float64)

    blocks = df["Address"].apply(
        lambda add: add[add.find("Block of ") + len("Block of "):] if add.find("Block of ") >= 0 else "NA")
    blocksLabel = preprocessing.LabelEncoder()
    blocksLabel.fit(blocks)
    df["blocks"] = blocksLabel.transform(blocks)

    normalize(df, ["X", "Y", "PdDistrict", "blocks"])


def normalize(df, cols):
    print("normalizing fields"+(', '.join(cols)))
    for col in cols:
        df[col] = preprocessing.scale(df[col].astype(numpy.float64))


if __name__ == "__main__":
    loc_train = "./data/train.csv"
    loc_test = "./data/test.csv"
    loc_submission = "./data/submission.csv"

    print("reading train data")
    df_train = pd.read_csv(loc_train)
    improveDate(df_train)
    improveAddress(df_train)
    print("label for category")

    categoryLabel = preprocessing.LabelEncoder()
    categoryLabel.fit(df_train["Category"])
    df_train["Category"] = categoryLabel.transform(df_train["Category"])

    # print("reading test data")
    # df_test = pd.read_csv(loc_test)
    # improveDate(df_test)
    # improveAddress(df_test)

    # descLabel = preprocessing.LabelEncoder()
    # descLabel.fit(df_train["Descript"])
    # df_train["Descript"] = preprocessing.scale(descLabel.transform(df_train["Descript"]).astype(numpy.float64))
    #
    # resolutionLabel = preprocessing.LabelEncoder()
    # resolutionLabel.fit(df_train["Resolution"])
    # df_train["Resolution"] = preprocessing.scale(resolutionLabel.transform(df_train["Resolution"]).astype(numpy.float64))



    feature_cols = [col for col in df_train.columns if
                    col not in ['Dates', 'Category', 'Descript', 'DayOfWeek', 'Address', 'Resolution']]
    X_train = df_train[feature_cols]
    y = df_train['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, y, test_size=0.33, random_state=42)

    # X_test = df_test[feature_cols]
    #
    clf = ensemble.RandomForestClassifier(n_estimators=25, n_jobs=-1)
    print("training data...")
    clf.fit(X_train, Y_train)

    print("data trained, predict on cross validation data")
    predicted = clf.predict(X_test)

    # print(categoryLabel.classes_)
    # print(predicted)
    # print("-----------------------")
    # print(Y_test)
    # print("-----------------------")
    # print(y)
    # print("-----------------------")
    print("prediction done, report stats:")
    print(metrics.classification_report(Y_test, predicted, target_names=categoryLabel.classes_))
    #
    # with open(loc_submission, "w") as outfile:
    # outfile.write("Id,Cover_Type\n")
    # for e, val in enumerate(list(clf.predict(X_test))):
    #     outfile.write("%s,%s\n"%(test_ids[e],val))

    # print("-----------------------")
    # print(X_train)
    # print("-----------------------")
    # print(X_test)
    #print(preprocessing.scale(df_train["blocks"].astype(numpy.float64)))