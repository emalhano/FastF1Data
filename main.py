# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:14:10 2021

@author: eduardob
"""

# Importing
import fastf1 as ff1
import fastf1.plotting
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
#from IPython import get_ipython

# Setup plotting
#plotting.setup_mpl()
#get_ipython().run_line_magic('matplotlib', 'qt')
#%matplotlib inline

bSelectLaps = False

# Enable the cache
ff1.Cache.enable_cache('cache')

# Get rid of some pandas warnings that are not relevant for us at the moment
pd.options.mode.chained_assignment = None

# Load the session data
race = ff1.get_session(2023, 'Miami', 'Qualifying')

# Get the laps
race.load()

drivers = ['SAI', 'PER']#, 'PER', 'VER', 'NOR', 'RUS','ALO','HAM', 'VET','OCO']#, 'PER', 'HAM', 'RUS', 'NOR', 'OCO', 'BOT', 'TSU']
colors = {'VER': 'b',        'PER': 'b',
          'HAM': 'black',    'RUS': 'black',
          'LEC': 'r',        'SAI': 'r',
          'NOR': 'orange',   'RIC': 'orange',
          'ALO': 'magenta',  'OCO': 'magenta',
          'GAS': 'darkblue', 'TSU': 'darkblue',
          'VET': 'green',    'STR': 'green', 'HUL': 'green',
          'BOT': 'gray',     'ZHO': 'gray',
          'ALB': 'darkblue', 'LAT': 'darkblue',
          'MSC': 'gray',     'MAG': 'gray'}

linestyles = {'VER': 'solid', 'PER': 'dashed',
              'HAM': 'solid', 'RUS': 'dashed',
              'LEC': 'solid', 'SAI': 'dashed',
              'NOR': 'solid', 'PIA': 'dashed',
              'OCO': 'solid', 'GAS': 'dashed',
              'TSU': 'solid', 'DEV': 'dashed',
              'ALO': 'solid', 'STR': 'dashed',
              'BOT': 'solid', 'ZHO': 'dashed',
              'ALB': 'solid', 'SAR': 'dashed',
              'MAG': 'solid', 'HUL': 'dashed'}

# Get laps of the drivers (HAM and VER)
laps_list = dict()
fastest = dict()
telemetry = dict()
for driver in drivers:
    laps_list[driver] = race.laps.pick_driver(driver)
    if not bSelectLaps:
        fastest[driver] = laps_list[driver].pick_fastest()
    else:
        if False: #driver == 'ALOsss':
            LapTimeList = laps_list[driver]['LapTime'].astype('timedelta64[ms]').astype(np.float64) / 1000
            LapNumberList = laps_list[driver]['LapNumber']
            LapTable = pd.DataFrame({'LapNumber': LapNumberList, 'LapTimeList': LapTimeList})
            print(LapTable)
            lapnum = input('Select lap number for ' + driver)
            fastest[driver] = laps_list[driver][laps_list[driver]['LapNumber'] == int(lapnum)]
            telemetry[driver] = fastest[driver].get_car_data().add_distance()
            #fastest[driver] = laps_list[driver].iloc[laps_list[driver].pick_fastest()
        else:
            LapTimeList = laps_list[driver]['LapTime'].astype('timedelta64[ms]').astype(np.float64) / 1000
            LapNumberList = laps_list[driver]['LapNumber']
            LapTable = pd.DataFrame({'LapNumber': LapNumberList, 'LapTimeList': LapTimeList})
            print(LapTable.to_string())
            lapnum = input('Select lap number for ' + driver)
            fastest[driver] = laps_list[driver][laps_list[driver]['LapNumber'] == int(lapnum)].pick_fastest()
            telemetry[driver] = fastest[driver].get_telemetry()#.get_car_data().add_distance()#
    telemetry[driver] = fastest[driver].get_telemetry()
    telemetry[driver]['tLap'] = telemetry[driver]['Time']/np.timedelta64(1, 's')#.astype('timedelta64[ms]').astype(np.float64) / 1000

# Get interpolated data so we can create TDiff channel
sLapInterp = np.linspace( 0., telemetry[drivers[0]]['Distance'].iloc[-1], 1000 )
sLapInterpDouble = np.linspace( 0., 2*telemetry[drivers[0]]['Distance'].iloc[-1], 2000 )
vCar = dict()
nGear = dict()
tLapDouble = dict()
tLap = dict()
rThrottle = dict()
GapToCarAhead = dict()
GapToCarAheadDouble = dict()
for driver in drivers:

    if driver == 'SAI':
        telemetry[driver]['Distance'] += 10
        telemetry[driver]['tLap'] += .158
        vCar[driver] = np.interp( sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['Speed'] )
        tLap[driver] = np.interp( sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['tLap'] )
        #tLap[driver] = fastest[driver].LapTime.total_seconds()*tLap[driver]/tLap[driver][-1]
        rThrottle[driver] = np.interp( sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['Throttle'] )
    else:
        telemetry[driver]['Distance'] = sLapInterp[-1] * telemetry[driver]['Distance'] / telemetry[driver]['Distance'].iloc[-1]
        vCar[driver] = np.interp(sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['Speed'])
        tLap[driver] = np.interp(sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['tLap'])
        tLap[driver] = fastest[driver].LapTime.total_seconds() * tLap[driver] / tLap[driver][-1]
        rThrottle[driver] = np.interp(sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['Throttle'])
        nGear[driver] = np.interp(sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['nGear'])
    tLapDouble[driver] = tLap[driver]
    tLapDouble[driver] = np.append(tLapDouble[driver], tLapDouble[driver] + tLapDouble[driver][-1] + 1e-3)
    GapToCarAhead[driver] = np.interp( sLapInterp, telemetry[driver]['Distance'], telemetry[driver]['DistanceToDriverAhead'] )

    GapToCarAheadDouble[driver] = GapToCarAhead[driver]
    GapToCarAheadDouble[driver] = np.append(GapToCarAhead[driver], GapToCarAheadDouble[driver])
    dt = np.interp(sLapInterpDouble + GapToCarAheadDouble[driver], sLapInterpDouble, tLapDouble[driver]) - tLapDouble[driver]

    GapToCarAhead[driver] = np.interp(sLapInterp, sLapInterpDouble, dt)




fig, ax = plt.subplots(4, sharex='all')


for ii, driver in enumerate(drivers):
    ax[0].plot(sLapInterp, vCar[driver], label=driver, color=colors[driver], linewidth=1, linestyle=linestyles[driver])
ax[0].set(ylabel='vCar [km/h]')
ax[0].grid(linewidth=0.5)

for ii, driver in enumerate(drivers):
    ax[1].plot(sLapInterp, rThrottle[driver], label=driver, color=colors[driver], linewidth=1, linestyle=linestyles[driver])
ax[1].set(ylabel='rPedalPos [%]')
ax[1].grid(linewidth=0.5)

for ii, driver in enumerate(drivers):
    ax[2].plot(sLapInterp, tLap[driver] - tLap[drivers[0]], label=driver, color=colors[driver], linewidth=1, linestyle=linestyles[driver])
ax[2].set(ylabel='TDiff [s]')
ax[2].legend(loc="lower left")
ax[2].grid(linewidth=0.5)

#for ii, driver in enumerate(drivers):
#    ax[2].plot(sLapInterp, nGear[driver], label=driver, color=colors[driver], linewidth=1, linestyle=linestyles[driver])
#ax[2].set(ylabel='nGear [-]')
#ax[2].legend(loc="lower left")
#ax[2].grid(linewidth=0.5)


for ii, driver in enumerate(drivers):
    ax[3].plot(sLapInterp, GapToCarAhead[driver], label=driver, color=colors[driver], linewidth=0.5, linestyle=linestyles[driver])
ax[3].set(ylabel='Gap to car ahead [s]', xlabel='sLap [m]')
ax[3].grid(linewidth=0.5)
#ax[3].set_ylim(0, 5.0)


#plt.suptitle('NOR - 77.218s | ALO - 81.452s', y=0.93)

plt.show()
