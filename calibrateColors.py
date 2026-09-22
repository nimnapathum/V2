#!/usr/bin/env python3

from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import ColorSensor
from time import sleep

# Color sensor connected to Port 1
color_sensor = ColorSensor(INPUT_1)

# Reflected light intensity mode
color_sensor.mode = 'COL-REFLECT'

print("Color sensor calibration")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        intensity = color_sensor.reflected_light_intensity

        print("Reflected light intensity:", intensity)

        sleep(0.3)

except KeyboardInterrupt:
    print("\nCalibration stopped.")