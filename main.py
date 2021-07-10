import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    return pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'],
                     nrows=500000)


def ridership_by_day(df):
    """ Plot by taxi drives by each day of week """
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_week'].plot.hist(bins=np.arange(8)-0.5,
                                ec='black',
                                ylim=(60000,75000))
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.show()


def ridership_by_hour(df):
    """ Plot by taxi drives by hour """
    df['hour'] = df['pickup_datetime'].dt.hour
    df['hour'].plot.hist(bins=24,
                         ec='black')
    plt.title('Pickup Hour Histogram')
    plt.xlabel('Hour')
    plt.show()


if __name__ == '__main__':
    df = load_data()
    ridership_by_day(df)
    ridership_by_hour(df)