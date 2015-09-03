#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import pandas as pd
import numpy as np
from quatrotate import qv_mult


class Load:

    """

    Process raw smartphone sensor data into a common format.

    Input
        dfloc: location on disk of csv file with raw smartphone data

    Output
        ndfloc: location on disk of new csv file with processed smartphone data

    """

    def __init__(self, dfloc):
        self.dfloc = dfloc
        self.process()

    def process(self):

        print("Reading csv file into Pandas dataframe")
        self.df = pd.read_csv(self.dfloc)

        # SensorLog app for iPhone
        if 'motionUserAccelerationX' in self.df.columns:
            self.app = 'SensorLog'

            # make a new column with date and time and set it as index
            self.df['DateTime'] = pd.DatetimeIndex(self.df['loggingTime'])
            self.df = self.df.set_index('DateTime')

            # drop unneeded columns
            unneeded = ['loggingTime', 'loggingSample', 
                    'identifierForVendor', 'deviceID', 
                    'locationTimestamp_since1970', 'locationVerticalAccuracy',
                    'locationHorizontalAccuracy', 'locationFloor',
                    'locationHeadingTimestamp_since1970',
                    'locationHeadingAccuracy',
                    'accelerometerTimestamp_sinceReboot',
                    'gyroTimestamp_sinceReboot', 'motionTimestamp_sinceReboot',
                    'motionMagneticFieldCalibrationAccuracy',
                    'activityTimestamp_sinceReboot', 'activity',
                    'activityActivityConfidence', 'activityActivityStartDate',
                    'pedometerStartDate', 'pedometerNumberofSteps',
                    'pedometerDistance', 'pedometerFloorAscended',
                    'pedometerFloorDescended', 'pedometerEndDate', 'IP_en0',
                    'IP_pdp_ip0', 'deviceOrientation', 'batteryState',
                    'batteryLevel', 'state', 'locationCourse',
                    'locationHeadingX', 'locationHeadingY', 'locationHeadingZ',
                    'locationTrueHeading', 'locationMagneticHeading',
                    'motionRotationRateX', 'motionRotationRateY',
                    'motionRotationRateZ', 'motionAttitudeReferenceFrame']
            self.df = self.df.drop(unneeded, axis=1)

            # rename remaining columns to standard system
            # assuming orientationZ = motionYaw, orientationY = motionRoll
            self.df.columns = ['latitude', 'longitude', 'altitude', 'speed',
                    'accelerometerX', 'accelerometerY', 'accelerometerZ', 
                    'gyroscopeX', 'gyroscopeY', 'gyroscopeZ', 
                    'orientationZ', 'orientationY', 'orientationX', 
                    'userAccelerationX', 'userAccelerationY', 
                    'userAccelerationZ',
                    'quaternionX', 'quaternionY', 'quaternionZ', 'quaternionW', 
                    'gravityX', 'gravityY', 'gravityZ', 
                    'magneticFieldX', 'magneticFieldY', 'magneticFieldZ']

        # AndroSensor app for Android
        if 'GYROSCOPE X (rad/s)' in self.df.columns:
            self.app = 'AndroSensor'

            # modify the android date column to be SS.SSS (vs. SS:SSS)
            if self.df['YYYY-MO-DD HH-MI-SS_SSS'][0][-4] == ':':
                print("Rewriting date column")
                reviseddates = []
                for i in range(len(self.df)):
                    start = self.df['YYYY-MO-DD HH-MI-SS_SSS'][i][0:-4]
                    mid = '.'
                    finish = self.df['YYYY-MO-DD HH-MI-SS_SSS'][i][-3:]
                    replaced = start + mid + finish
                    reviseddates.append(replaced)
                self.df['YYYY-MO-DD HH-MI-SS_SSS'] = reviseddates

            print("Reformatting date column to DatetimeIndex")
            # make a new column with date and time and set it as index
            self.df['DateTime'] = \
                    pd.DatetimeIndex(self.df['YYYY-MO-DD HH-MI-SS_SSS'])
            self.df = self.df.set_index('DateTime')

            print("Dropping unnecessary columns")
            # drop unneeded columns
            unneeded = ['LIGHT (lux)', 'PROXIMITY (i)', 
                    'ATMOSPHERIC PRESSURE (hPa)', 'SOUND LEVEL (dB)',
                    'LOCATION Altitude-google ( m)', 
                    'LOCATION Altitude-atmospheric pressure ( m)',
                    'LOCATION Accuracy ( m)', 'LOCATION ORIENTATION (°)',
                    'Satellites in range', 'Time since start in ms ',
                    'YYYY-MO-DD HH-MI-SS_SSS']
            self.df = self.df.drop(unneeded, axis=1)

            print("Renaming remaining columns")
            # rename remaining columns to standard system
            self.df = self.df.rename(columns={\
                    'ACCELEROMETER X (m/s²)':'accelerometerX',
                    'ACCELEROMETER Y (m/s²)':'accelerometerY',
                    'ACCELEROMETER Z (m/s²)':'accelerometerZ',
                    'GRAVITY X (m/s²)':'gravityX',
                    'GRAVITY Y (m/s²)':'gravityY',
                    'GRAVITY Z (m/s²)':'gravityZ',
                    'LINEAR ACCELERATION X (m/s²)':'userAccelerationX',
                    'LINEAR ACCELERATION Y (m/s²)':'userAccelerationY',
                    'LINEAR ACCELERATION Z (m/s²)':'userAccelerationZ',
                    'GYROSCOPE X (rad/s)':'gyroscopeX',
                    'GYROSCOPE Y (rad/s)':'gyroscopeY',
                    'GYROSCOPE Z (rad/s)':'gyroscopeZ',
                    'MAGNETIC FIELD X (μT)':'magneticFieldX',
                    'MAGNETIC FIELD Y (μT)':'magneticFieldY',
                    'MAGNETIC FIELD Z (μT)':'magneticFieldZ',
                    'ORIENTATION Z (azimuth °)':'orientationZ',
                    'ORIENTATION X (pitch °)':'orientationX',
                    'ORIENTATION Y (roll °)':'orientationY',
                    'LOCATION Latitude : ':'latitude',
                    'LOCATION Longitude : ':'longitude',
                    'LOCATION Altitude ( m)':'altitude',
                    'LOCATION Speed ( Kmh)':'speed'})

            print(self.df.columns)
            print("Computing quaternion")
            # compute the Quaternion 4-vector and add it to the dataframe
            qw, qx, qy, qz = self.getQuat()
            self.df['quaternionW'] = qw
            self.df['quaternionX'] = qx
            self.df['quaternionY'] = qy
            self.df['quaternionZ'] = qz

        # rotate the XYZ measurements to a common reference frame
        print("Rotating to common reference frame...")
        self.rotate()

        # resample to 10 Hz (period = 100 ms)
        print("Resampling to 10 Hz")
        self.df.resample('100L')

        # write resulting dataframe to a new csv file
        self.write()


    def getQuat(self):

        """ Given 3 orientation angles, compute the quaternion. """

        yaw = self.df['orientationZ'] / 2. * np.pi / 180
        roll = self.df['orientationX'] / 2. * np.pi / 180
        pitch = self.df['orientationY'] / 2. * np.pi / 180

        w =  np.cos(roll) * np.cos(pitch) * np.cos(yaw) + \
                np.sin(roll) * np.sin(pitch) * np.sin(yaw)

        x =  np.sin(roll) * np.cos(pitch) * np.cos(yaw) - \
                np.cos(roll) * np.sin(pitch) * np.sin(yaw)

        y =  np.cos(roll) * np.sin(pitch) * np.cos(yaw) + \
                np.sin(roll) * np.cos(pitch) * np.sin(yaw)

        z =  np.cos(roll) * np.cos(pitch) * np.sin(yaw) - \
                np.sin(roll) * np.sin(pitch) * np.cos(yaw)

        return w, x, y, z

    def rotate(self):

        """ Generate rdf, a rotated version of df where the z-axis is aligned
        with gravity. """

        varlist = ['accelerometer', 'orientation', 'userAcceleration',
                'gravity', 'magneticField', 'gyroscope']

        quaternion = self.df[['quaternionW', 'quaternionX', 
            'quaternionY', 'quaternionZ']].values

        for ivar in varlist:
            print("..." + ivar)
            xyzlist = [ivar + 'X', ivar + 'Y', ivar + 'Z']
            xyz = self.df[xyzlist].values
            xyz_rotated = getrot(quaternion, xyz)
            self.df[ivar + 'X'] = xyz_rotated[:, 0]
            self.df[ivar + 'Y'] = xyz_rotated[:, 1]
            self.df[ivar + 'Z'] = xyz_rotated[:, 2]

    def write(self):

        """ Write dataframe to disk. """

        ndfloc = self.dfloc[0:-4] + '_processed.csv'
        print("Storing dataframe with modified date column to " +
                ndfloc)
        self.df.to_csv(ndfloc)

def getrot(quatern, vector):
    rotatedvector = []
    for i in range(vector.shape[0]):
        rotatedvector.append(qv_mult(tuple(quatern[i, :]), 
            tuple(vector[i, :])))
    return np.array(rotatedvector)
