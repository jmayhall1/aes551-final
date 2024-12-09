# coding=utf-8
"""
@author: John Mark Mayhall & Therese Parkes
Last Edit: 11/17/2024
email: jmm0111@uah.edu
"""
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import glob
from datetime import datetime, timedelta

import warnings


def pgf_plotter(direction: list, path: str) -> None:
    """
    Function for plotting the average PGF magnitude for the state of Colorado.
    :param direction: List of min/max lat/lon values.
    :param path: String to data files.
    :return: Nothing.
    """
    warnings.filterwarnings("ignore")  # Removes Warnings
    north, south, east, west = direction  # Sets min/max lat/lon
    height_dict = {0: '1000hPa', 1: '925hPa', 2: '850hPa', 3: '700hPa', 4: '600hPa', 5: '500hPa',
                   6: '400hPa', 7: '300hPa'}
    pgf_storage = pd.DataFrame(columns=['1000hPa', '925hPa', '850hPa', '700hPa', '600hPa', '500hPa',
                                        '400hPa', '300hPa', 'Date', 'Time'])  # Creates empty dictionary
    file_list = glob.glob(path)  # Grabs data files

    start = datetime(1800, 1, 1, 0, 0)  # This is the "days since" part

    data = Dataset(file_list[0])  # Loads in file
    average_lat = (north + south) / 2
    f = 2 * 7.292 * (10 ** -5) * np.sin(np.radians(average_lat))  # Coriolis parameter
    for k in range(data.variables['hgt'].shape[0]):
        delta = timedelta(hours=int(data.variables['time'][k]))  # Create a time delta object from the number of days
        offset = start + delta  # Add the specified number of days to 1990

        date = offset.strftime('%Y%m%d')  # Gets date
        time = offset.strftime('%H%M')
        lons = data.variables['lon'][:] - 180  # Gets lons
        east_index, west_index = (np.abs(lons - east)).argmin(), (np.abs(lons - west)).argmin()  # Creates index slices
        lons = lons[west_index: east_index + 1]  # Slices lons

        lats = data.variables['lat'][:]  # Gets lats
        north_index, south_index = ((np.abs(lats - north)).argmin(),
                                    (np.abs(lats - south)).argmin())  # creates index slices
        lats = lats[north_index: south_index + 1]  # Slices lats

        lon, lat = np.meshgrid(lons, lats)  # Creates lat/lon grid
        geo_data = data.variables['hgt'][k, :, north_index: south_index + 1,
                   west_index: east_index + 1]  # Gets Pressure
        temp_data_storage = []  # Empty list for data storage

        for i in range(np.shape(geo_data)[0]):
            if i == 8:
                break
            geo = np.array(geo_data[i])  # Gets pressure for specific time and height

            average_x = np.average(geo, axis=1)  # Averages pressure into single longitude slice
            n = len(average_x)  # Gets length of pressure longitude slice
            dx = (6378000 * np.cos((np.radians(np.average(lat, axis=0)[0] + np.average(lat, axis=0)[-1]) / 2)) *
                  (np.radians(np.average(lon, axis=0)[-1] - np.average(lon, axis=0)[0])))
            dpress_dx = (np.sum(average_x * np.arange(n - 1, -n, -2)) / (n * (n - 1) / 2) / dx)  # Calculates east-west pressure gradient

            average_y = np.average(geo, axis=0)  # Averages pressure into single latitude slice
            n = len(average_y)  # Gets length of pressure latitude slice
            dy = 6378000 * np.radians((np.average(lat, axis=1)[0] -
                                       np.average(lat, axis=1)[-1]))  # Calculates north-south length
            dpress_dy = ((np.sum(average_y * np.arange(n - 1, -n, -2)) /
                          (n * (n - 1) / 2) / dy))  # Calculates north-south pressure gradient
            u, v = -dpress_dy / f, dpress_dx / f
            bearing = np.abs(np.degrees(np.arctan(v / u)))  # Calculates PGF initial bearing
            if u > 0 and v > 0:
                degree = bearing  # Sets degree if quadrant 1
            if u < 0 < v:
                degree = 180 - bearing  # Sets degree if quadrant 2
            if u < 0 and v < 0:
                degree = 180 + bearing  # Sets degree if quadrant 3
            if u > 0 > v:
                degree = 360 - bearing  # Sets degree if quadrant 4
            temp_data_storage.append(degree)  # Stores PGF degree
        temp_data_storage.append(date)  # Stores date
        temp_data_storage.append(time)  # Stores time
        temp_data_storage = pd.DataFrame(temp_data_storage).T  # Converts list to dataframe
        temp_data_storage.columns = ['1000hPa', '925hPa', '850hPa', '700hPa', '600hPa', '500hPa',
                                     '400hPa', '300hPa', 'Date', 'Time']  # Sets dataframe columns
        pgf_storage = pd.concat((pgf_storage, temp_data_storage), ignore_index=True)  # Concatenates dataframes

    plt.rcParams["figure.figsize"] = (9, 5)  # Sets figure length

    '''Averages yearly data to create an average PGF diurnal plot and resets storage dataframe accordingly'''
    zero_utc = pgf_storage.loc[pgf_storage['Time'] == '0000'].drop(columns=['Time', 'Date']).mean()
    six_utc = pgf_storage.loc[pgf_storage['Time'] == '0600'].drop(columns=['Time', 'Date']).mean()
    twelve_utc = pgf_storage.loc[pgf_storage['Time'] == '1200'].drop(columns=['Time', 'Date']).mean()
    eighteen_utc = pgf_storage.loc[pgf_storage['Time'] == '1800'].drop(columns=['Time', 'Date']).mean()
    pgf_storage = pd.DataFrame(columns=['1000hPa', '925hPa', '850hPa', '700hPa', '600hPa', '500hPa',
                                        '400hPa', '300hPa', 'Date', 'Time'])
    zero_utc = pd.DataFrame(zero_utc).T
    zero_utc['Time'] = '0000 UTC'
    six_utc = pd.DataFrame(six_utc).T
    six_utc['Time'] = '0600 UTC'
    twelve_utc = pd.DataFrame(twelve_utc).T
    twelve_utc['Time'] = '1200 UTC'
    eighteen_utc = pd.DataFrame(eighteen_utc).T
    eighteen_utc['Time'] = '1800 UTC'
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(zero_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(six_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(twelve_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(eighteen_utc)))

    '''All lines below, until the next comment, plot the PGF magnitude for each model pressure level'''
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['1000hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (1000hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_1000.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['925hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (925hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_925.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['850hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (850hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_850.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['700hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (700hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_700.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['600hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (600hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_600.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['500hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (500hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_500.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['400hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (400hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_400.jpg')
    plt.show()
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['300hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time (300hPa)')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('//uahdata/rstor/aes551_project_new/geo_degree_plots/geo_dir_300.jpg')
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    pgf_plotter(direction=[45, 35, -100, -110],
                path='//uahdata/rstor/aes551_project_new/data/*.nc')  # Runs the code