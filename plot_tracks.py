#!/usr/bin/env python3

import sys
import os
import re
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt


def init_args_parser():

    parser = argparse.ArgumentParser(
        description='Extrack tracks states from Log file and plot the their trajectory')

    parser.add_argument('-t', '--track_id', type=int, nargs='+', required=False,
                        help='tracks id that will be plotted')

    parser.add_argument('-st', '--start_timestamp', type=float, required=False, default=0,
                        help='start timestamp for plot. default is 0')

    parser.add_argument('-d', '--duration', type=float, required=False, default=-1,
                        help='duration for plot. default is 1000')

    return parser

#  track: id 201 T 1684306653.794 states 29.039 -0.012 7.130 -0.020 -0.083 0.010 yaw -0.002 yaw_rate -0.009 size 4.503 2.101 1.795 category Sedan prob 0.942 center_w 848900.456 2522280.538 32.882 vel_w -2.386 6.719 0.016 acc_w 0.016 -0.082 -0.074 cov 0.044 0.044 0.156 0.156 0.362 0.362 yaw_3d 1.913 roll_3d 0.000 pitch_3d 0.001 age 2.328


class TrackTrajectory:
    def __init__(self, id):
        self.id = id
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.ax = []
        self.ay = []
        self.az = []
        self.l = []
        self.w = []
        self.h = []
        self.yaw = []
        self.yaw_rate = []

    def append(self, timestamp, states_list):
        if not states_list:
            return

        self.t.append(timestamp)
        self.x.append(float(states_list[1]))
        self.y.append(float(states_list[2]))
        self.z.append(float(states_list[3]))
        self.vx.append(float(states_list[4]))
        self.vy.append(float(states_list[5]))
        self.vz.append(float(states_list[6]))
        self.ax.append(float(states_list[7]))
        self.ay.append(float(states_list[8]))
        self.az.append(float(states_list[9]))

        self.l.append(float(states_list[10]))
        self.w.append(float(states_list[11]))
        self.h.append(float(states_list[12]))
        self.yaw.append(float(states_list[13]))
        self.yaw_rate.append(float(states_list[14]))

    def plot(self, output_path):
        figure, ax = plt.subplots(3, 4, figsize=(25, 10), dpi=200)
        t = np.array(self.t) % 10000
        ax[0][0].plot(t, self.x)
        ax[0][1].plot(t, self.y)
        ax[0][2].plot(t, self.z)
        ax[1][0].plot(t, self.vx)
        ax[1][1].plot(t, self.vy)
        ax[1][2].plot(t, self.vz)
        ax[2][0].plot(t, self.ax)
        ax[2][1].plot(t, self.ay)
        ax[2][2].plot(t, self.az)

        ax[0][0].set_ylabel("x/m")
        ax[0][1].set_ylabel("y/m")
        ax[0][2].set_ylabel("z/m")
        ax[1][0].set_ylabel("vx/m/s")
        ax[1][1].set_ylabel("vy/m/s")
        ax[1][2].set_ylabel("vz/m/s")
        ax[2][0].set_ylabel("ax/m/s2")
        ax[2][1].set_ylabel("ay/m/s2")
        ax[2][2].set_ylabel("az/m/s2")

        ax[0][3].plot(t, self.yaw)
        ax[1][3].plot(t, self.yaw_rate)
        ax[0][3].set_ylabel("yaw/rad")
        ax[1][3].set_ylabel("yaw_rate/rad/s")

        for r in range(3):
            for c in range(4):
                ax[r][c].grid()
                ax[r][c].set_xlabel("t/s")

        figure.suptitle("Track id:"+str(self.id)+" Trajectory")
        filename = output_path+"track_"+str(self.id)+".png"
        figure.savefig(filename)
        print("Save track states plot as", filename)


class TracksExtractor:

    def __init__(self):
        self.tracks = []
        self.tracks_id = []

    def extract_tracks(self, log_filename):

        with open(log_filename, "r") as log_f:
            lines = log_f.readlines()
            k = 0
            while k < len(lines):
                states_list = lines[k].split()
                if len(states_list) == 3:
                    frame_index = int(states_list[0])
                    tracks_num = int(states_list[1])
                    timestamp = float(states_list[2])
                    for i in range(tracks_num):
                        k = k + 1
                        states_list = lines[k].split()
                        id = int(states_list[0])
                        if id in self.tracks_id:
                            idx = self.tracks_id.index(id)
                            self.tracks[idx].append(timestamp, states_list)
                        else:
                            self.tracks_id.append(id)
                            self.tracks.append(TrackTrajectory(id))
                            self.tracks[-1].append(timestamp, states_list)
                k = k + 1
            print("Collect", len(self.tracks), "tracks")

    def plot_tracks(self, output_path):

        for track in self.tracks:
            if len(track.t) > 30:
                track.plot(output_path)


def get_log_file_path(root_path):
    files_name = os.listdir(root_path)
    latest_time = 0
    latest_file = None
    for file_name in files_name:
        time = int(file_name[-14:-4])
        if (latest_time < time):
            latest_time = time
            latest_file = file_name
    return latest_file


def get_plots_path(log_filename):

    path = "./plots/" + log_filename[:-4] + "/"
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create plots path:", path)
        return path
    return None


def main():

    root_path = "./multi_sensor_mot/results/"
    log_filename = get_log_file_path(root_path)
    filename = root_path + log_filename

    tracks_extractor = TracksExtractor()
    tracks_extractor.extract_tracks(filename)
    tracks_extractor.plot_tracks(get_plots_path(log_filename))


if __name__ == '__main__':

    main()
