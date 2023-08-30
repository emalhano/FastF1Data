import fastf1 as ff1
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import fastf1.plotting

#fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# Get rid of some pandas warnings that are not relevant for us at the moment
pd.options.mode.chained_assignment = None


driver_to_team_2022 = {"VER": "red bull", "PER": "red bull",
                       "HAM": "mercedes", "RUS": "mercedes",
                       "LEC": "ferrari", "SAI": "ferrari",
                       "OCO": "alpine", "ALO": "alpine",
                       "NOR": "mclaren", "RIC": "mclaren",
                       "VET": "aston martin", "STR": "aston martin",
                       "BOT": "alfa romeo", "ZHO": "alfa romeo",
                       "GAS": "alphatauri", "TSU": "alphatauri",
                       "MAG": "haas", "MSC": "haas",
                       "ALB": "williams", "LAT": "williams"}

driverlinestyle = {"VER": "solid", "PER": "dashed",
                   "HAM": "solid", "RUS": "dashed",
                   "LEC": "solid", "SAI": "dashed",
                   "ALO": "solid", "OCO": "dashed",
                   "NOR": "solid", "RIC": "dashed",
                   "VET": "solid", "STR": "dashed",
                   "BOT": "solid", "ZHO": "dashed",
                   "GAS": "solid", "TSU": "dashed",
                   "MAG": "solid", "MSC": "dashed",
                   "ALB": "solid", "LAT": "dashed"}

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
        lap_data["Speed"] = 3.6 * np.gradient(lap_data["Distance"], lap_data["Time"])
        lap_data["gLong"] = np.gradient(np.array(lap_data["Speed"]), np.array(lap_data["Time"])) / 3.6
        lap_data["is_PLS"] = lap_data["Throttle"] > 98
        lap_data["is_GLS"] = lap_data["Throttle"] < 98
        lap_data["gLong_brake"] = lap_data["gLong"] * (lap_data["gLong"] < 0)
        lap_data["DRS"] = lap_data["DRS"] > 9

        time_DRS = np.trapz(np.array(lap_data["DRS"].astype(np.float64)), x=np.array(lap_data['Time'])) / lap_time
        time_GLS = np.trapz(np.array(lap_data["is_GLS"].astype(np.float64)), x=np.array(lap_data['Time'])) / lap_time
        time_PLS = np.trapz(np.array(lap_data["is_PLS"].astype(np.float64)), x=np.array(lap_data['Time'])) / lap_time
        PBrake = np.trapz(-np.array(lap_data["gLong_brake"] * lap_data["Speed"] / 3.6),
                          x=np.array(lap_data['Time'])) * 800 / 1e3 / lap_time

        return time_DRS, time_GLS, time_PLS, PBrake

    return 0., 0., 0., 0.


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

    metrics = {"track": [], "time_DRS": [], "time_GLS": [], "time_PLS": [], "PBrake": []}
    for track in tracks:
        time_DRS, time_GLS, time_PLS, PBrake = get_track_insights(track)
        metrics["track"].append(track)
        metrics["time_DRS"].append(time_DRS)
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

    season_metrics.sort_values("time_DRS", inplace=True)
    fig = plt.figure()
    plt.barh(season_metrics["track"], season_metrics["time_DRS"], color='r')
    plt.xlabel("% of lap with DRS")
    plt.grid()

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
                     "nGear": [],
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
        lap_local["nGear"] = np.array(lap_df["nGear"])
        lap_local["Throttle"] = np.array(lap_df["Throttle"])
        lap_local["DRS"] = np.array(lap_df["DRS"])
        lap_local["DRS"] = lap_local["DRS"] > 9
        lap_local["DistanceToDriverAhead"] = np.array(lap_df["DistanceToDriverAhead"])
        lap_local["Distance_interp"] = sLapInterp

        tLapDouble = lap_local["Time_interp"]
        tLapDouble = np.append(lap_local["Time_interp"], lap_local["Time_interp"] + lap_local["Time_interp"][-1] + 1e-3)
        lap_local["GapToCarAhead"] = np.interp(sLapInterp, lap_local["Distance"], lap_local["DistanceToDriverAhead"])

        GapToCarAheadDouble = lap_local["GapToCarAhead"]
        GapToCarAheadDouble = np.append(GapToCarAheadDouble, GapToCarAheadDouble)
        dt = np.interp(sLapInterpDouble + GapToCarAheadDouble, sLapInterpDouble, tLapDouble) - tLapDouble

        lap_local["GapToCarAhead"] = np.interp(sLapInterp, sLapInterpDouble, dt)

        lap_data.append(lap_local)

    fig, ax = plt.subplots(3, sharex='all')
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
    axDRS.set(ylabel='DRS [on/off]', ylim=(0, 5))
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

        #ax[2].plot(lap_data[ii]["Distance"], lap_data[ii]["nGear"] - 0*lap_data[ii]["Distance"],
        #           label=legend_name, color=color, linewidth=1)
    #ax[2].set(ylabel='nGear [-]', xlabel='Distance [m]')
    #ax[2].legend(loc="lower center")
    #ax[2].grid(linewidth=0.5)

    for ii, driver in enumerate(lap_dict["driver"]):

        if use_color:
            color = lap_dict["color"][ii]
        else:
            color = fastf1.plotting.driver_color(driver)

        if use_legend:
            legend_name = lap_dict["legend_name"][ii]
        else:
            legend_name = driver

        ax[2].plot(lap_data[ii]["Distance"], lap_data[ii]["nGear"],# - lap_data[0]["Time_interp"],
                   label=legend_name, color=color, linewidth=1)
    ax[2].set(ylabel='nGear [-]', xlabel='sLap [m]', ylim=(0, 10))
    ax[2].grid(linewidth=0.5)
    ax[2].legend(loc="upper left")

    plt.show()


def plot_race_strategy(track_name, year):

    session = ff1.get_session(year, track_name, "Race")
    session.load()

    drivers = session.laps["Driver"].unique().tolist()

    number_of_laps = session.laps["LapNumber"].max()
    final_laps = session.laps.loc[session.laps["LapNumber"] == number_of_laps]
    race_winner = final_laps.loc[final_laps["Time"] == final_laps["Time"].min()]["Driver"].to_string().split('    ')[1]

    red_flag_timestamp = session.race_control_messages["Time"].loc[
        session.race_control_messages["Message"] == "RED FLAG"].to_list()
    safety_car_deployed_timestamp = session.race_control_messages["Time"].loc[
        session.race_control_messages["Message"] == "SAFETY CAR DEPLOYED"].to_list()
    safety_car_in_timestamp = session.race_control_messages["Time"].loc[
        session.race_control_messages["Message"] == "SAFETY CAR IN THIS LAP"].to_list()

    red_flagged_laps = []
    for red_flag in red_flag_timestamp:
        red_flag_dt_to_each_lap = (session.laps.pick_driver(race_winner).LapStartDate - red_flag).dt.total_seconds()
        red_flagged_laps.append(
            session.laps["LapNumber"][red_flag_dt_to_each_lap.loc[red_flag_dt_to_each_lap < 0].index[-1]])

    safety_car_deployed_laps = []
    for safety_car_out in safety_car_deployed_timestamp:
        safety_car_dt_to_each_lap = (
                    session.laps.pick_driver(race_winner).LapStartDate - safety_car_out).dt.total_seconds()
        safety_car_deployed_laps.append(
            session.laps["LapNumber"][safety_car_dt_to_each_lap.loc[safety_car_dt_to_each_lap < 0].index[-1]])

    safety_car_in_laps = []
    for safety_car_in in safety_car_in_timestamp:
        safety_car_dt_to_each_lap = (
                    session.laps.pick_driver(race_winner).LapStartDate - safety_car_in).dt.total_seconds()
        safety_car_in_laps.append(
            session.laps["LapNumber"][safety_car_dt_to_each_lap.loc[safety_car_dt_to_each_lap < 0].index[-1]])


    n_race_splits = len(red_flag_timestamp) + 1

    lap_deltas = {key: np.array([]) for key in drivers}
    for ii in range(0, n_race_splits):

        if ii == 0 and n_race_splits > 1:
            laps_split = session.laps.loc[session.laps["LapNumber"] <= red_flagged_laps[ii]]
        elif ii == n_race_splits - 1 and n_race_splits > 1:
            laps_split = session.laps.loc[session.laps["LapNumber"] > red_flagged_laps[ii - 1]]
        elif n_race_splits > 1:
            laps_split = session.laps.loc[red_flagged_laps[ii - 1] < session.laps["LapNumber"] <= red_flagged_laps[ii]]
        else:
            laps_split = session.laps.loc[session.laps["LapNumber"] > 0]

        number_of_laps_in_split = len(laps_split.pick_driver(race_winner))
        average_lap_race_winner = (laps_split.pick_driver(race_winner)["Time"].iloc[-1] -
                                   laps_split.pick_driver(race_winner)["LapStartTime"].iloc[
                                       0]).total_seconds() / number_of_laps_in_split
        race_winner_delta_time = np.array(
            [average_lap_race_winner * lap_number for lap_number in range(1, number_of_laps_in_split + 1)])

        for driver in drivers:
            for jj in range(0, number_of_laps_in_split):
                if jj < len(laps_split.pick_driver(driver)):
                    lap_deltas[driver] = np.append(lap_deltas[driver], race_winner_delta_time[jj] - (
                                laps_split.pick_driver(driver).iloc[jj]["Time"] -
                                laps_split.pick_driver(driver).iloc[0]["LapStartTime"]).total_seconds())


    for driver in drivers:
        if year == 2023:
            color = fastf1.plotting.driver_color(driver)
            line_style = "solid"
        else:
            color = fastf1.plotting.TEAM_COLORS[driver_to_team_2022[driver]]
            line_style = driverlinestyle[driver]
        plt.plot(range(1, len(lap_deltas[driver])+1), lap_deltas[driver], linestyle=line_style, label=driver, color=color, zorder=2)

    for ii, safety_car_deployed_lap in enumerate(safety_car_deployed_laps):
        plt.axvspan(safety_car_deployed_lap, safety_car_in_laps[ii], facecolor='yellow', zorder=3)

    plt.grid(zorder=1)
    plt.xlabel("Lap number [-]")
    plt.ylabel("Gap to winner's average lap")
    plt.legend(loc='upper left', ncol=int(len(drivers)/2))
    plt.show()


if __name__ == "__main__":
    # get_season_insights()
    # plot_season_metrics()
    plot_race_strategy("Netherlands", 2022)

    drivers = ["LEC"]


    compare_laps({"year": [2022 for driver in drivers],
                  "track": ["Netherlands" for driver in drivers],
                  "session": ["Qualifying" for driver in drivers],
                  "driver": [driver for driver in drivers],
                  "lap": [0],
                  "legend_name": [driver for driver in drivers]})
