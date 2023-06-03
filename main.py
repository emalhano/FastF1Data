import fastf1 as ff1
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import fastf1.plotting


fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# Get rid of some pandas warnings that are not relevant for us at the moment
pd.options.mode.chained_assignment = None

def get_track_insights(track_name):

    # Load the session data
    session = ff1.get_session(2022, track_name, 'Qualifying')
    session.load(weather=True)

    is_dry = not session.weather_data["Rainfall"].any()

    get_metrics = False
    if is_dry:
        get_metrics = True
    else:
        # Load the session data
        session = ff1.get_session(2022, track_name, 'FP2')
        session.load(weather=True)

        is_dry = not session.weather_data["Rainfall"].any()

        if is_dry:
            get_metrics = True


    if get_metrics:

        # Get fastest lap
        fastest_lap = session.laps.pick_fastest()
        lap_time = fastest_lap["LapTime"] / np.timedelta64(1, 's')

        lap_data = fastest_lap.get_telemetry()
        lap_data["Time"] = lap_data["Time"] / np.timedelta64(1, 's')
        lap_data["Speed"] = 3.6*np.gradient(lap_data["Distance"], lap_data["Time"])
        lap_data["gLong"] = np.gradient(np.array(lap_data["Speed"]), np.array(lap_data["Time"]))/3.6
        lap_data["is_PLS"] = lap_data["Throttle"] > 98
        lap_data["is_GLS"] = lap_data["Throttle"] < 98
        lap_data["gLong_brake"] = lap_data["gLong"]*(lap_data["gLong"] < 0)

        time_GLS = np.trapz(np.array(lap_data["is_GLS"].astype(np.float64)), x=np.array(lap_data['Time'])) / lap_time
        time_PLS = np.trapz(np.array(lap_data["is_PLS"].astype(np.float64)), x=np.array(lap_data['Time'])) / lap_time
        PBrake   = np.trapz(-np.array(lap_data["gLong_brake"]*lap_data["Speed"]/3.6), x=np.array(lap_data['Time']))*800/1e3 / lap_time

        return time_GLS, time_PLS, PBrake

    return 0., 0., 0.




def get_season_insights():

    tracks = ["Bahrain",
              "Jeddah",
              "Melbourne",
              "Imola",
              "Miami",
              "Barcelona",
              "Monaco",
              "Baku",
              "Montreal",
              "Silverstone",
              "Austria",
              "France",
              "Hungary",
              "Spa",
              "Zandvoort",
              "Monza",
              "Singapore",
              "Suzuka",
              "Austin",
              "Mexico",
              "Interlagos",
              "Abu Dhabi"]

    metrics = {"track": [], "time_GLS": [], "time_PLS": [], "PBrake": []}
    for track in tracks:
        time_GLS, time_PLS, PBrake = get_track_insights(track)
        metrics["track"].append(track)
        metrics["time_GLS"].append(time_GLS)
        metrics["time_PLS"].append(time_PLS)
        metrics["PBrake"].append(PBrake)

    season_metrics = pd.DataFrame(metrics)

    if not os.path.exists("Results"):
        os.mkdir("Results")
    season_metrics.to_csv("Results/season_metrics.csv")




def plot_season_metrics():

    try:
        season_metrics = pd.read_csv("Results/season_metrics.csv")
    except FileNotFoundError as e:
        print("No file for season metrics found. Please run get_season_insights() first.")
        print(e)
        return

    season_metrics.sort_values("time_GLS", inplace=True)
    fig = plt.figure()
    plt.barh(season_metrics["track"], season_metrics["time_GLS"], color='r')
    plt.xlabel("% of lap grip limited")
    plt.grid()

    season_metrics.sort_values("time_PLS", inplace=True)
    fig = plt.figure()
    plt.barh(season_metrics["track"], season_metrics["time_PLS"], color='r')
    plt.xlabel("% of lap power limited")
    plt.grid()

    season_metrics.sort_values("PBrake", inplace=True)
    fig = plt.figure()
    plt.barh(season_metrics["track"], season_metrics["PBrake"], color='r')
    plt.xlabel("Average estimated brake power [kW]")
    plt.grid()
    plt.show()




def compare_laps(lap_dict, channels=None):

    try:
        assert (len(lap_dict["track"]) ==
                len(lap_dict["year"]) ==
                len(lap_dict["session"]) ==
                len(lap_dict["driver"]) ==
                len(lap_dict["lap"]))

        use_color = False
        if "color" in lap_dict.keys():
            assert (len(lap_dict["track"]) == len(lap_dict["color"]))
            use_color = True

        use_legend = False
        if "legend_name" in lap_dict.keys():
            assert (len(lap_dict["track"]) == len(lap_dict["legend_name"]))
            use_legend = True

    except AssertionError as e:
        print("Sizes of dictionary entries don't match.")
        print(e)
        return


    lap_data = []
    for ii, driver in enumerate(lap_dict["driver"]):

        session = ff1.get_session(lap_dict["year"][ii], lap_dict["track"][ii], lap_dict["session"][ii])
        session.load()
        laps = session.laps.pick_driver(driver)

        if lap_dict["lap"][ii] == 0:
            lap_df = laps.pick_fastest().get_telemetry()
            lap_time = laps.pick_fastest().LapTime.total_seconds()
        else:
            lap_df = laps.iloc[lap_dict["lap"][ii]].get_telemetry()
            lap_time = laps.iloc[lap_dict["lap"][ii]].LapTime.total_seconds()

        lap_df['Time'] = lap_df['Time'].dt.total_seconds()

        if ii == 0:
            lap_distance = lap_df["Distance"].iloc[-1]
            sLapInterp = np.linspace(0., lap_df["Distance"].iloc[-1], 1000)
            sLapInterpDouble = np.linspace(0., 2 * lap_df["Distance"].iloc[-1], 2000)
            tLapRef = np.interp(sLapInterp, lap_df['Distance'], lap_df['Time'])

        lap_local = {"Distance": [],
                     "Time_interp": [],
                     "Speed": [],
                     "Throttle": [],
                     "DRS": [],
                     "Distance_interp": [],
                     "DistanceToDriverAhead": [],
                     "GapToCarAhead": []}

        lap_local["Distance"] = np.array(lap_df["Distance"] * sLapInterp[-1] / lap_df["Distance"].iloc[-1])
        lap_local["Time_interp"] = np.interp(sLapInterp, lap_local["Distance"], lap_df['Time'])
        lap_local["Time_interp"] = lap_local["Time_interp"] * lap_time / lap_local["Time_interp"][-1]

        lap_local["Speed"] = np.array(lap_df["Speed"])
        lap_local["Throttle"] = np.array(lap_df["Throttle"])
        lap_local["DRS"] = np.array(lap_df["DRS"])
        lap_local["DRS"] = lap_local["DRS"] > 9
        lap_local["DistanceToDriverAhead"] = np.array(lap_df["DistanceToDriverAhead"])
        lap_local["Distance_interp"] = sLapInterp


        tLapDouble = lap_local["Time_interp"]
        tLapDouble = np.append(lap_local["Time_interp"], lap_local["Time_interp"] + lap_local["Time_interp"][-1] + 1e-3)
        lap_local["GapToCarAhead"] = np.interp( sLapInterp, lap_local["Distance"], lap_local["DistanceToDriverAhead"] )

        GapToCarAheadDouble = lap_local["GapToCarAhead"]
        GapToCarAheadDouble = np.append(GapToCarAheadDouble, GapToCarAheadDouble)
        dt = np.interp(sLapInterpDouble + GapToCarAheadDouble, sLapInterpDouble, tLapDouble) - tLapDouble

        lap_local["GapToCarAhead"] = np.interp(sLapInterp, sLapInterpDouble, dt)

        lap_data.append(lap_local)


    fig, ax = plt.subplots(4, sharex='all')
    axDRS = ax[0].twinx()

    for ii, driver in enumerate(lap_dict["driver"]):

        if use_color:
            color = lap_dict["color"][ii]
        else:
            color = fastf1.plotting.driver_color(driver)

        if use_legend:
            legend_name = lap_dict["legend_name"][ii]
        else:
            legend_name = driver

        ax[0].plot(lap_data[ii]["Distance"], lap_data[ii]["Speed"], label=legend_name,
                   color=color, linewidth=1)
        axDRS.plot(lap_data[ii]["Distance"], lap_data[ii]["DRS"], label=legend_name,
                   color=color, linewidth=1)
    ax[0].set(ylabel='vCar [km/h]')
    axDRS.set(ylabel='DRS [on/off]', ylim=(0,5))
    ax[0].grid(linewidth=0.5)

    for ii, driver in enumerate(lap_dict["driver"]):

        if use_color:
            color = lap_dict["color"][ii]
        else:
            color = fastf1.plotting.driver_color(driver)

        if use_legend:
            legend_name = lap_dict["legend_name"][ii]
        else:
            legend_name = driver

        ax[1].plot(lap_data[ii]["Distance"], lap_data[ii]["Throttle"], label=legend_name,
                   color=color, linewidth=1)
    ax[1].set(ylabel='rPedalPos [%]')
    ax[1].grid(linewidth=0.5)

    for ii, driver in enumerate(lap_dict["driver"]):

        if use_color:
            color = lap_dict["color"][ii]
        else:
            color = fastf1.plotting.driver_color(driver)

        if use_legend:
            legend_name = lap_dict["legend_name"][ii]
        else:
            legend_name = driver

        ax[2].plot(lap_data[ii]["Distance_interp"], lap_data[ii]["Time_interp"] - lap_data[0]["Time_interp"],
                   label=legend_name, color=color, linewidth=1)
    ax[2].set(ylabel='tDiff [s]')
    ax[2].legend(loc="lower left")
    ax[2].grid(linewidth=0.5)

    for ii, driver in enumerate(lap_dict["driver"]):

        if use_color:
            color = lap_dict["color"][ii]
        else:
            color = fastf1.plotting.driver_color(driver)

        if use_legend:
            legend_name = lap_dict["legend_name"][ii]
        else:
            legend_name = driver

        ax[3].plot(lap_data[ii]["Distance_interp"], lap_data[ii]["GapToCarAhead"], label=legend_name,
                   color=color, linewidth=1)
    ax[3].set(ylabel='Gap to car ahead [s]', xlabel='sLap [m]', ylim=(0, 10))
    ax[3].grid(linewidth=0.5)

    plt.show()



if __name__ == "__main__":

    #plot_season_metrics()
    compare_laps({"year": [2023, 2023],
                  "track": ["Barcelona", "Barcelona"],
                  "session": ['Qualifying', 'Qualifying'],
                  "driver": ['GAS', 'HAM'],
                  "lap": [0,0]})

