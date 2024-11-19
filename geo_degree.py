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

import warnings


def geo_direction_plotter(direction: list, path: str, degree=None) -> None:
    """
    Function for plotting the average geostrophic wind direction for the state of Colorado.
    :param degree: Placeholder for degree value and can be changed to use a different value for missing degrees.
    :param direction: List of min/max lat/lon values.
    :param path: String to data files.
    :return: Nothing.
    """
    warnings.filterwarnings("ignore")  # Removes Warnings
    north, south, east, west = direction  # Sets min/max lat/lon
    average_lat = (north + south) / 2
    heights = [71, 65, 59, 55, 52, 49, 47, 44]  # Sets height indices
    time_dict = {0: '0000 UTC', 1: '0300 UTC', 2: '0600 UTC', 3: '0900 UTC', 4: '1200 UTC',
                 5: '1500 UTC', 6: '1800 UTC', 7: '2100 UTC'}  # Creates dictionary for time assignment
    pgf_storage = pd.DataFrame(columns=['985hPa', '895hPa', '800hPa', '700hPa', '600hPa', '487.5hPa', '412.5hPa',
                                        '288.083hPa', 'Date', 'Time'])  # Creates empty dictionary
    file_list = glob.glob(path)  # Grabs data files
    f = 2 * 7.292 * (10 ** -5) * np.sin(np.radians(average_lat))  # Coriolis parameter
    for k, file in enumerate(file_list):
        print(f'Processing file {k + 1} of {len(file_list)}')
        data = Dataset(file)  # Loads in file
        date = file[-12: -4]  # Gets date
        lons = data.variables['lon'][:]  # Gets lons
        east_index, west_index = (np.abs(lons - east)).argmin(), (np.abs(lons - west)).argmin()  # Creates index slices
        lons = lons[west_index: east_index]  # Slices lons

        lats = data.variables['lat'][:]  # Gets lats
        north_index, south_index = ((np.abs(lats - north)).argmin(),
                                    (np.abs(lats - south)).argmin())  # creates index slices
        lats = lats[south_index: north_index]  # Slices lats

        lon, lat = np.meshgrid(lons, lats)  # Creates lat/lon grid
        press_data = data.variables['PL'][:, :, south_index: north_index, west_index: east_index]  # Gets Pressure
        temperature_data = data.variables['T'][:, :, south_index: north_index,
                           west_index: east_index]  # Gets Temperature

        for i in range(np.shape(press_data)[0]):
            time = time_dict.get(i)  # Gets time
            temp_data_storage = []  # Empty list for data storage

            for j in heights:
                press = press_data[i][j]  # Gets pressure for specific time and height
                temperature = temperature_data[i][j]  # Gets temperature for specific time and height
                density = np.average(press) / (np.average(temperature) * 287)  # Calculates density

                average_x = np.average(press, axis=1)  # Averages pressure into single longitude slice
                n = len(average_x)  # Gets length of pressure longitude slice
                dx = (6378000 * np.cos((np.average(lat, axis=0)[0] + np.average(lat, axis=0)[-1]) / 2) *
                      (np.average(lon, axis=0)[-1] - np.average(lon, axis=0)[0]))  # Calculates east-west length
                dpress_dx = -((np.sum(average_x * np.arange(n - 1, -n, -2)) / (n * (n - 1) / 2) / dx) /
                              density)  # Calculates east-west pressure gradient

                average_y = np.average(press, axis=0)  # Averages pressure into single latitude slice
                n = len(average_y)  # Gets length of pressure latitude slice
                dy = 6378000 * (np.average(lat, axis=1)[-1] -
                                np.average(lat, axis=1)[0])  # Calculates north-south length
                dpress_dy = -((np.sum(average_y * np.arange(n - 1, -n, -2)) / (n * (n - 1) / 2) / dy) /
                              density)  # Calculates north-south pressure gradient
                u, v = dpress_dy / f, -dpress_dx / f
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
            temp_data_storage.columns = ['985hPa', '895hPa', '800hPa', '700hPa', '600hPa', '487.5hPa',
                                         '412.5hPa', '288.083hPa', 'Date', 'Time']  # Sets dataframe columns
            pgf_storage = pd.concat((pgf_storage, temp_data_storage), ignore_index=True)  # Concatenates dataframes

    plt.rcParams["figure.figsize"] = (9, 5)  # Sets figure size

    '''Creates scatter plot of PGF directions'''
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['985hPa']))), label='PGF at 985hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['895hPa']))), label='PGF at 895hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['800hPa']))), label='PGF at 800hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['700hPa']))), label='PGF at 700hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['600hPa']))), label='PGF at 600hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['487.5hPa']))), label='PGF at 487.5hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['412.5hPa']))), label='PGF at 412.5hPa')
    plt.scatter(pgf_storage.Time, np.array(list(map(float, pgf_storage['288.083hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.legend(loc='upper right')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_degree_scatter.jpg')
    plt.close('all')

    '''Averages yearly data to create an average PGF diurnal plot and resets storage dataframe accordingly'''
    zero_utc = pgf_storage.loc[pgf_storage['Time'] == '0000 UTC'].drop(columns=['Time', 'Date']).mean()
    three_utc = pgf_storage.loc[pgf_storage['Time'] == '0300 UTC'].drop(columns=['Time', 'Date']).mean()
    six_utc = pgf_storage.loc[pgf_storage['Time'] == '0600 UTC'].drop(columns=['Time', 'Date']).mean()
    nine_utc = pgf_storage.loc[pgf_storage['Time'] == '0900 UTC'].drop(columns=['Time', 'Date']).mean()
    twelve_utc = pgf_storage.loc[pgf_storage['Time'] == '1200 UTC'].drop(columns=['Time', 'Date']).mean()
    fifteen_utc = pgf_storage.loc[pgf_storage['Time'] == '1500 UTC'].drop(columns=['Time', 'Date']).mean()
    eighteen_utc = pgf_storage.loc[pgf_storage['Time'] == '1800 UTC'].drop(columns=['Time', 'Date']).mean()
    twentyone_utc = pgf_storage.loc[pgf_storage['Time'] == '2100 UTC'].drop(columns=['Time', 'Date']).mean()
    pgf_storage = pd.DataFrame(columns=['985hPa', '895hPa', '800hPa', '700hPa', '600hPa', '487.5hPa', '412.5hPa',
                                        '288.083hPa', 'Date', 'Time'])
    zero_utc = pd.DataFrame(zero_utc).T
    zero_utc['Time'] = '0000 UTC'
    three_utc = pd.DataFrame(three_utc).T
    three_utc['Time'] = '0300 UTC'
    six_utc = pd.DataFrame(six_utc).T
    six_utc['Time'] = '0600 UTC'
    nine_utc = pd.DataFrame(nine_utc).T
    nine_utc['Time'] = '0900 UTC'
    twelve_utc = pd.DataFrame(twelve_utc).T
    twelve_utc['Time'] = '1200 UTC'
    fifteen_utc = pd.DataFrame(fifteen_utc).T
    fifteen_utc['Time'] = '1500 UTC'
    eighteen_utc = pd.DataFrame(eighteen_utc).T
    eighteen_utc['Time'] = '1800 UTC'
    twentyone_utc = pd.DataFrame(twentyone_utc).T
    twentyone_utc['Time'] = '2100 UTC'
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(zero_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(three_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(six_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(nine_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(fifteen_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(eighteen_utc)))
    pgf_storage = pd.concat((pgf_storage, pd.DataFrame(twentyone_utc)))

    '''All lines below, until the next comment, plot the PGF direction for each model pressure level'''
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['985hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_985_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['895hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_895_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['800hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_800_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['700hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_700_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['600hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_600_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['487.5hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_487_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['412.5hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_412_degree.jpg')
    plt.close('all')

    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['288.083hPa']))))
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_288_degree.jpg')
    plt.close('all')

    '''Plots all model level PGF average directions together in one line plot'''
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['985hPa']))), label='PGF at 985hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['895hPa']))), label='PGF at 895hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['800hPa']))), label='PGF at 800hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['700hPa']))), label='PGF at 700hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['600hPa']))), label='PGF at 600hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['487.5hPa']))), label='PGF at 487.5hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['412.5hPa']))), label='PGF at 412.5hPa')
    plt.plot(pgf_storage.Time, np.array(list(map(float, pgf_storage['288.083hPa']))), label='PGF at 288.083hPa')
    plt.title('Average Geostrophic Wind Direction vs Time')
    plt.ylabel(r'Direction ($\degree$)')
    plt.xlabel('Time (UTC)')
    plt.legend(loc='upper right')
    plt.savefig('/rstor/jmayhall/aes551_project/geo_degree_plots/geo_total_degree.jpg')


if __name__ == '__main__':
    geo_direction_plotter(direction=[41, 37, -102, -109],
                          path='/rstor/jmayhall/aes551_project/data/*.nc4')  # Runs the code
