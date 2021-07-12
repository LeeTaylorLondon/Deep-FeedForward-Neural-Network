import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def load_data():
    """ Load 500,000 records due to memory limits """
    return pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'],
                     nrows=500000)

def ridership_by_day(df):
    """ Data exloration - Plot by taxi drives by each day of week """
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_week'].plot.hist(bins=np.arange(8)-0.5,
                                ec='black',
                                ylim=(60000,75000))
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.show()

def ridership_by_hour(df):
    """ Data exloration - Plot by taxi drives by hour """
    df['hour'] = df['pickup_datetime'].dt.hour
    df['hour'].plot.hist(bins=24,
                         ec='black')
    plt.title('Pickup Hour Histogram')
    plt.xlabel('Hour')
    plt.show()

def clean_data(df, verbose=0):
    """ Clean data by removing missing values and outliers """
    if verbose: print(df.isnull().sum())
    df = df.dropna() # delete rows with NaN value(s)
    # Cleaning fare prices
    if verbose:
        print(df.describe())
        df['fare_amount'].hist(bins=500)
        plt.xlabel("Fare")
        plt.title("Histogram of Fares")
        plt.show()
    df = df[(df['fare_amount'] >=0) & (df['fare_amount'] <= 100)]
    # Passenger count
    if verbose:
        df['passenger_count'].hist(bins=6, ec='black')
        plt.xlabel("Passenger Count")
        plt.title("Histogram of Passenger Count")
        plt.show()
    # replace 0 passenger count with 1 (set 0s to 1s)
    df.loc[df['passenger_count']==0, 'passenger_count'] = 1
    if verbose:
        # Shows some coordinates are out of Earth's range
        df.plot.scatter('pickup_longitude', 'pickup_latitude')
        plt.show()
    # longitude range for NYC
    nyc_min_longitude = -74.05
    nyc_max_longitude = -73.75
    # latitude range for NYC
    nyc_min_latitude = 40.63
    nyc_max_latitude = 40.85
    # data only contains journeys in NYC
    for long in ['pickup_longitude', 'dropoff_longitude']:
        df = df[(df[long] > nyc_min_longitude) &
                (df[long] < nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        df = df[(df[lat] > nyc_min_latitude) &
                (df[lat] < nyc_max_latitude)]
    return df

def euc_distance(lat1, long1, lat2, long2):
    """ Returns the distance between two points
     used in function data engineering """
    return ((lat1-lat2)**2 + (long1-long2)**2)**0.5

def data_engineering(df, verbose=0):
    if verbose: print(df.head()['pickup_datetime'])
    # convert datetime into machine readable categories
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour
    if verbose: print(df.loc[:5,['pickup_datetime,', 'year', 'month',
                                 'day', 'day_of_week', 'hour']])
    df = df.drop(['pickup_datetime'],axis=1)
    # new column distance
    df['distance'] = euc_distance(df['pickup_latitude'],
                                  df['pickup_longitude'],
                                  df['dropoff_latitude'],
                                  df['dropoff_longitude'])
    if verbose:
        df.plot.scatter('fare_amount', 'distance')
        plt.show()
    # airports have fixed fares this affects results and learning
    # create new column for the distance away from airports
    airports = {'JFK_Airport':(-73.78, 40.643),
                'Laguarida_Airport':(-73.87, 40.77),
                'Newark_Airport':(-74.18, 40.69)}
    for airport in airports:
        df['pickup_dist_' + airport] = euc_distance(df['pickup_latitude'],
                                                df['pickup_longitude'],
                                                airports[airport][1],
                                                airports[airport][0])
        df['dropoff_dist_' + airport] = euc_distance(df['dropoff_latitude'],
                                                     df['dropoff_longitude'],
                                                     airports[airport][1],
                                                     airports[airport][0])
    if verbose:
        print(df[['key', 'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude',
                  'pickup_dist_JFK_Airport',
                  'dropoff_dist_JFK_Airport']].head())
    df = df.drop(['key'], axis=1)
    return df

def data_preprocessing(df):
    df_prescaled = df.copy() # original copy of data
    # remove fare_amount as we do not want to scale these values
    df_scaled = df.drop(['fare_amount'], axis=1)
    # features in df_scaled are scaled
    df_scaled = scale(df_scaled)
    # columns are coverted to a list
    cols = df.columns.tolist()
    # ensure fare_amount is removed
    cols.remove('fare_amount')
    # set df_scaled to cols converted to pd.DataFrame type
    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
    # concatenate fare_amount to scaled features
    df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
    # over write df
    df = df_scaled.copy()
    return df, df_prescaled

def split_data(df):
    x = df.loc[:, df.columns != 'fare_amount']
    y = df.loc[:, 'fare_amount']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    return x_train, x_test, y_train, y_test

def build_model(input_shape, verbose=False):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    if verbose: model.summary()
    model.compile(loss="mse", optimizer='adam', metrics=['mse', 'accuracy'])
    return model

def test_model(model, *data):
    if len(data) != 4: raise TypeError('Incorrect amount of data passed to test_model(...).')
    x_train, x_test, y_train, y_test = data
    train_pred = model.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Train RMSE: {:0.2f}".format(train_rmse))
    print("Test RMSE: {:0.2f}".format(test_rmse))


if __name__ == '__main__':
    df = load_data()
    """ Data exploration """
    # ridership_by_day(df)
    # ridership_by_hour(df)
    """ Data cleaning, engineering, preprocessing """
    df = clean_data(df)
    df = data_engineering(df)
    df, _ = data_preprocessing(df)
    """ Model, building, learning, and testing """
    x_train, x_test, y_train, y_test = split_data(df)
    model = build_model(x_train.shape[1])
    model.fit(x_train, y_train, epochs=1)
    test_model(model, x_train, x_test, y_train, y_test)