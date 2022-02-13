import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def preprossingdata(filename, filepath, test_size=0.2, unused_col=[], encode_col=[]):
    df = pd.read_csv(filepath + filename)
    # print(df.head())
    df_copy = df.copy()
    print(df_copy.head())
    col_names = df_copy.columns

    last = df_copy[[col_names[-1]]]
    print(last.nunique())
    print(last.value_counts())
    # last.value_counts().plot(kind='bar', title='Categories vs Counts')
    last.value_counts().plot(kind='pie', title='Categories vs Counts',y='Counts')

    plt.tight_layout()
    plt.savefig(filename+'Data Distribution.png')
    if encode_col !=None:
        for i in encode_col:
            labelset = list(set(df_copy.iloc[:,i]))
            le = LabelEncoder()
            le.fit(labelset)
            df_copy.iloc[:,i] = le.transform(df_copy.iloc[:,i])
            # print(x_data.iloc[:,0])

    # discard the unique ID columns
    feature_cols = col_names[0:-1]
    if unused_col != None:
        feature_cols = feature_cols.delete(unused_col)
    class_col = col_names[-1]
    x_data = df_copy[feature_cols]
    y_data = df_copy[class_col]


    print(x_data.columns)
    print(x_data.shape)
    print(y_data.shape)


    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

    print(x_train.shape)
    print(x_test.shape)

    # Normalize training and testing data separately
    stdscaler = StandardScaler()
    stdscaler.fit(x_train)
    x_train = stdscaler.transform(x_train)
    stdscaler.fit(x_test)
    x_test = stdscaler.transform(x_test)

    return x_train, x_test, y_train, y_test