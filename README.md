# sensor-fusion
Use accelerometer and gyroscope data from smartphones to identify vehicle type (bus or car) and phone location (driver side or passenger side).

  1. Extract gravity signal (see "Extract Gravity Signal" jupyter notebook)
  2. Rotate XYZ signals to vehicle reference frame (see "Rotate Sensor Data to
  Vehicle Reference Frame" jupyter notebook)
  3. Resample time series data to 10 Hz sampling rate (see "Resample Sensor
  Data to 10 Hz Sampling Rate" jupyter notebook)
  4. Automate steps 1-3, modify columns to standard system, and save result to
  a new file (see "Process Smartphone Sensor Data" jupyter notebook)
