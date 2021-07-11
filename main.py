import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
                                  df['pickup_longtitude'],
                                  df['dropoff_latitude'],
                                  df['dropoff_longtitude'])
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


if __name__ == '__main__':
    df = load_data()
    """ Data exploration """
    # ridership_by_day(df)
    # ridership_by_hour(df)
