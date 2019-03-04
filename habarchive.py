import sys
import os
import pathlib
import glob
import xml.dom.minidom as md
from pathlib import Path
import kml2geojson as kg
import json
import geojson
from folium import plugins
from folium.plugins import HeatMap
import numpy as np
from datetime import datetime, timedelta, time
import statistics
import matplotlib.pyplot as plt
import datetime
import time
import pandas as pd
from geopy.distance import geodesic
import math
import csv
from scipy.optimize import curve_fit


mission_dist = []
mission_bear = []
mission_alt = []
mission_ids = []
ID = []
mean_ascent = []
mean_descent = []
descent_a = []
descent_aerr = []
descent_b = []
descent_berr = []

def openCSV(missionID):
    event = []
    with open('./data/anasonde/abs%d.csv' % missionID, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    for x in range(len(data)):
        if x == 0:
            pass
        elif data[x][5].find("#") != -1:
            pass
        else:
            update = [data[x][5], data[x][6], data[x][7], data[x][8],
                      data[x][13], data[x][14], data[x][16], data[x][17]]
            # this strips relevant data from anasonde file. local time, longitude, latitude, gps altitude, pressure, temperature, humidity
            event.append(update)
    return event


def delta_time(t1, t2):
    FMT = '%H:%M:%S'
    deltat = datetime.datetime.strptime(
        t2, FMT) - datetime.datetime.strptime(t1, FMT)
    deltat = deltat.total_seconds()
    # print(deltat)
    return deltat


def alt_press(pressure, sealevel):
    altitude = 44330 * (1.0 - pow(float(pressure) / float(sealevel), 0.1903))
    return altitude


def heading(lat1, lon1, lat2, lon2):
    delLat = math.radians(lat2 - lat1)
    delLon = math.radians(lon2 - lon1)

    y = math.sin(delLon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.cos(delLat)
    # returns the bearing from true north
    tempBearing = math.degrees(math.atan2(y, x))
    while tempBearing < 0:      # Makes sure the bearing is between 0 and 360
        tempBearing += 360
    while tempBearing > 360:
        tempBearing -= 360
    return tempBearing


def processData(event, missionID, sealvlpress):
    alt_corr = []
    horz_speed = []
    distance = []
    ascent = []
    descent = []
    direction = []
    datapool = []
    bearings = []
    altitude = []
    alt2 = []


    for x in range(len(event) - 1):
        # print(event)

        elapsed_time = delta_time(event[x][0], event[x + 1][0])
        start_point = (event[x][1], event[x][2])
        end_point = (event[x + 1][1], event[x + 1][2])
        distance.append(geodesic(start_point, end_point).meters)
        if elapsed_time != 0:
            horz_speed.append(distance[x] / elapsed_time)
        else:
            horz_speed.append(0)
        direction.append(heading(float(event[x][1]), float(
            event[x][2]), float(event[x + 1][1]), float(event[x + 1][2])))
        land = end_point
        if x == 0:
            lon1 = float(event[x][2])
            lat1 = float(event[x][1])
            launch = (lat1, lon1)

        # print(event[x][3])
        lon2 = float(event[x][2])
        lat2 = float(event[x][1])
        bearings.append(heading(lat1, lon1, lat2, lon2))
        sheight = alt_press(event[x][4], sealvlpress)
        eheight = alt_press(event[x + 1][4], sealvlpress)
        rise = eheight - sheight
        if rise >= 0:
            if elapsed_time != 0:
                ascent.append(rise / elapsed_time)
            else:
                ascent.append(0)

        else:
            if elapsed_time != 0:
                descent.append(-rise / elapsed_time)
                alt2.append(sheight)

            else:
                descent.append(0)
                alt2.append(sheight)

        alt_corr.append(sheight)
        if elapsed_time != 0:
            rise_rate = rise / elapsed_time
        else:
            rise_rate = 0

        altitude.append(float(event[x][3]))

        proc_data = [missionID, event[x][0], float(event[x][1]), float(event[x][2]), float(event[x][3]), float(
            event[x][4]), float(event[x][5]), event[x][6], event[x][7], alt_corr[x], rise_rate, horz_speed[x], direction[x]]
        datapool.append(proc_data)
    mission_ids.append(missionID)

    ID.append(missionID)
    tot_dist = geodesic(launch, land).meters
    tot_bear = heading(lat1, lon1, lat2, lon2)
    mission_dist.append(tot_dist)
    mission_bear.append(tot_bear)

    max_alt = max(altitude)
    mission_alt.append(max_alt)

    mean_ascent.append(statistics.mean(ascent))
    
    descentplot = np.column_stack((alt2, descent))
    popt, pcov = descentAnalysis(alt2, descent)

    errs = np.sqrt(np.diag(pcov))
    descent_a.append(popt[0])
    descent_aerr.append(errs[0])
    descent_b.append(popt[1])
    descent_berr.append(errs[1])

    descent_df = pd.DataFrame(descentplot, columns=['Altitude', 'Speed'])
    descent_df.to_csv('results/abs%s-descent.csv' % missionID)
    writeExcel(missionID, datapool, ascent, descent)
    createKML(missionID, event)


def createKML(missionID, event):
    kmlPath = []

    kmlPath = [
        '<kml><Document><name>Arkansas BalloonSAT Mission %d</name><Style id="stratoLine">\n<LineStyle>\n<width>1.0</width>\n</LineStyle>\n</Style>\n<Placemark>\n<name>Simulation</name>\n<styleUrl>#stratoLine</styleUrl>\n<LineString>\n<coordinates>\n' % (
            missionID)]

    for x in range(len(event) - 1):
        #altitude = alt_press(event[x][3], '1023')
        kmlPath.append('%s,%s,%s\n' % (event[x][2], event[x][1], event[x][3]))

    kmlPath.append(
        '</coordinates>\n<altitudeMode>absolute</altitudeMode>\n</LineString>\n</Placemark>\n</Document>\n</kml>')

    kmlFile = "".join(kmlPath)
    with open('results/abs%d.kml' % missionID, 'w') as file:
        file.write(kmlFile)


def writeExcel(missionID, datapool, ascent, descent):

    labels = ['Mission ID', 'Local Time', 'Longitude', 'Latitude', 'Altitude', 'Pressure', 'Temperature',
              'Temperature2', 'Humidity', 'Altitude-corr', 'Vertical velocity', 'Horizontal Velocity', 'Horizontal Direction']

    df = pd.DataFrame(datapool, columns=labels)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('results/data_abs%d.xlsx' %
                            missionID, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='data')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def missionData():
    writer = pd.ExcelWriter('results/abs-missions.xlsx', engine='xlsxwriter')
    mission_df = pd.DataFrame([mission_ids, mission_bear, mission_dist, mission_alt, mean_ascent, descent_a, descent_aerr, descent_b, descent_berr], index=[
                              'Mission ID', 'Bearing', 'Distance', 'Max Altitude', 'Ascent Speed', 'Descent A', 'DA error', 'Descent B', 'DB error'])
    mission_df.T.to_excel(writer, sheet_name='highlights')
    writer.save()


def missionInit(missionID):
    file = r'./data/mission-pressure.xlsx'
    df = pd.read_excel(file)
    # print(df)
    x = df.loc[df['Mission ID'] == missionID]['SLP'].astype('float')
    y = x.iloc[0]
    # print(y)
    return y


def func(x, d, p):
    return np.sqrt(2 * d * 9.78 * np.exp(x / p))


def descentAnalysis(xdata, ydata):
    return curve_fit(func, xdata, ydata, p0=[1, 6000], bounds=((0, 0), (np.inf, np.inf)))
