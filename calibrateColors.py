#!/usr/bin/env python3
"""Simple EV3 reflected-light calibration helper."""
from time import sleep
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import ColorSensor
from ev3dev2.button import Button

sensor = ColorSensor(INPUT_1)
sensor.mode = 'COL-REFLECT'
btn = Button()


def wait_press():
    while btn.enter:
        sleep(0.05)
    while not btn.enter:
        sleep(0.05)
    while btn.enter:
        sleep(0.05)


def sample(label, count=20):
    print('\nPlace sensor over {} and press ENTER.'.format(label))
    wait_press()
    values = []
    for i in range(count):
        value = sensor.reflected_light_intensity
        values.append(value)
        print('{:02d}: {}'.format(i + 1, value))
        sleep(0.08)
    print('{} -> min={}, max={}, average={:.1f}'.format(
        label, min(values), max(values), sum(values) / len(values)))
    return values


print('EV3 COLOR SENSOR CALIBRATION')
print('Keep sensor at the exact height/orientation used during driving.')
black = sample('BLACK')
edge = sample('EDGE (boundary)')
white = sample('WHITE')

print('\nSUMMARY')
print('BLACK: min={} max={} avg={:.1f}'.format(min(black), max(black), sum(black)/len(black)))
print('EDGE : min={} max={} avg={:.1f}'.format(min(edge), max(edge), sum(edge)/len(edge)))
print('WHITE: min={} max={} avg={:.1f}'.format(min(white), max(white), sum(white)/len(white)))
print('\nChoose BLACK_MAX between the black range and edge range.')
print('Choose WHITE_MIN between the edge range and white range.')
