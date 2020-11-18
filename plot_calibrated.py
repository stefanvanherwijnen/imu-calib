import numpy as np
import argparse
import time

from code.helpers import *
from code.cost_functions import *
from code.utilities import *

import matplotlib.pyplot as plt
import json

np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run calibration on real data from IMU.')
    parser.add_argument('--sampling_frequency', help = 'Sampling frequency for logfile.', 
        required = True, type = int)
    parser.add_argument('--file', help = 'Path to file with data from IMU.',
        required = True, type = str)
    parser.add_argument('--calibration', help = 'Path to calibration file.',
        required = True, type = str)
    args = parser.parse_args()

    datafile = args.file
    calibrationfile = args.calibration
    dt = 1 / args.sampling_frequency

    # read file with ax, ay, az, wx, wy, wz measurements from IMU
    imu_data = np.genfromtxt(datafile, delimiter=' ')
    with open(calibrationfile) as json_file:  
      calibration = json.load(json_file)

    I = np.array([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0], 
                  [0.0, 0.0, 1.0]])

    Macc = calibration['acc']['M']
    Cacc = np.linalg.inv(I+Macc)
    bacc = calibration['acc']['b']
    calibrated_acc = np.copy(imu_data[:,0:3])
    for measurement in calibrated_acc:
        measurement[:] = Cacc @ (measurement - bacc)

    Mgyro = calibration['gyro']['M']
    bgyro = calibration['gyro']['b']
    Cgyro = np.linalg.inv(I+Mgyro)
    R = calibration['gyro']['R']
    Rinv = np.linalg.inv(R)
    calibrated_gyro = np.copy(imu_data[:,3:6])

    for measurement in calibrated_gyro:
        measurement[:] = Cgyro @ Rinv @ measurement - Cgyro @ bgyro

    plot_raw_and_calibrated_euler_angles(imu_data[:,3:6], calibrated_gyro, dt)

    Tgyro = np.array([[0.0005, 0.0, 0.0],
                    [0.0, 0.0005, 0.0],
                    [0.0, 0.0, 0.0005]])
    Tgyroinv = np.linalg.inv(Tgyro)
    distored_gyro = np.copy(imu_data[:,3:6])
    for measurement in distored_gyro:
        measurement[:] = Tgyro @ measurement
    fixed_gyro = np.copy(distored_gyro)
    for measurement in fixed_gyro:
        measurement[:] = Tgyroinv @ Cgyro @ Rinv @ measurement - Cgyro @ bgyro
    # plot_raw_and_calibrated_euler_angles(calibrated_gyro, fixed_gyro, dt)
    plot_calibrated_and_uncalibrated_acc_norms(imu_data[:,0:3], calibrated_acc)
    plt.show()