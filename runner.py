#!/usr/bin/env python3
"""Episode-based Q-learning edge follower for LEGO EV3."""
import json
import os
import random
from time import sleep

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.button import Button

# ---------------- Hardware ----------------
left_motor = LargeMotor(OUTPUT_D)
right_motor = LargeMotor(OUTPUT_A)
color_sensor = ColorSensor(INPUT_1)
ir_sensor = InfraredSensor(INPUT_4)
btn = Button()
color_sensor.mode = 'COL-REFLECT'

# ---------------- Calibration / motion ----------------
BASE_SPEED = 16
TURN_SPEED = 16
ACTION_TIME = 0.06
BLACK_MAX = 6       # CALIBRATE for your track
WHITE_MIN = 40      # CALIBRATE for your track
OBSTACLE_THRESHOLD = 22
Q_FILE = 'edge_q_table.json'

# ---------------- RL ----------------
NUM_STATES = 3
NUM_ACTIONS = 4
ACTION_NAMES = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Reverse'}
STATE_NAMES = {0: 'Black', 1: 'Edge', 2: 'White'}

ALPHA = 0.45
GAMMA = 0.90
START_EPSILON = 0.35
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.88
EPISODES = 15
MAX_STEPS = 300

Q_table = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]


def get_current_state():
    reflection = color_sensor.reflected_light_intensity
    if reflection < BLACK_MAX:
        return 0
    if reflection < WHITE_MIN:
        return 1
    return 2


def execute_action(action):
    if action == 0:  # forward
        left_motor.on(SpeedPercent(BASE_SPEED))
        right_motor.on(SpeedPercent(BASE_SPEED))
    elif action == 1:  # left pivot
        left_motor.on(SpeedPercent(-TURN_SPEED))
        right_motor.on(SpeedPercent(TURN_SPEED))
    elif action == 2:  # right pivot
        left_motor.on(SpeedPercent(TURN_SPEED))
        right_motor.on(SpeedPercent(-TURN_SPEED))
    elif action == 3:  # reverse
        left_motor.on(SpeedPercent(-20))
        right_motor.on(SpeedPercent(-20))


def stop_motors():
    left_motor.off()
    right_motor.off()


def get_reward(prev_state, action, next_state):
    # Edge is the target state.
    if next_state == 1:
        if action == 0:
            return 10
        if action in (1, 2):
            return 2 if prev_state == 1 else 8
        return -5

    # On black: right turn should recover this chosen edge orientation.
    if next_state == 0:
        if prev_state == 1:
            return -3 if action == 0 else -8
        if action == 0:
            return -10
        if action == 2:
            return 7
        return -6

    # On white: left turn should recover this chosen edge orientation.
    if prev_state == 1:
        return -3 if action == 0 else -8
    if action == 1:
        return 7
    return -3


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)

    # Random tie-breaking avoids an initial bias toward action 0.
    best_value = max(Q_table[state])
    best_actions = [a for a, value in enumerate(Q_table[state])
                    if value == best_value]
    return random.choice(best_actions)


def update_q(state, action, reward, next_state):
    best_next = max(Q_table[next_state])
    old_q = Q_table[state][action]
    Q_table[state][action] = old_q + ALPHA * (
        reward + GAMMA * best_next - old_q
    )


def save_q_table():
    with open(Q_FILE, 'w') as f:
        json.dump(Q_table, f)


def load_q_table():
    global Q_table
    if not os.path.exists(Q_FILE):
        return False
    try:
        with open(Q_FILE, 'r') as f:
            loaded = json.load(f)
        if len(loaded) != NUM_STATES or any(len(row) != NUM_ACTIONS for row in loaded):
            raise ValueError('Q-table dimensions do not match')
        Q_table = [[float(v) for v in row] for row in loaded]
        return True
    except Exception as exc:
        print('Could not load Q-table:', exc)
        return False


def wait_for_enter(message):
    print(message)
    # Require release first so one press cannot trigger twice.
    while btn.enter:
        sleep(0.05)
    while not btn.enter:
        sleep(0.05)
    while btn.enter:
        sleep(0.05)


def print_q_table():
    print('\n{:<8} {:>9} {:>9} {:>9} {:>9}'.format(
        'State', 'Forward', 'Left', 'Right', 'Reverse'))
    for s in range(NUM_STATES):
        print('{:<8} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}'.format(
            STATE_NAMES[s], *Q_table[s]))


def run_episode(episode, epsilon):
    """Run one bounded physical episode and return total reward."""
    total_reward = 0
    current_state = get_current_state()

    print('\n=== Episode {}/{} | epsilon={:.3f} | start={} ==='.format(
        episode, EPISODES, epsilon, STATE_NAMES[current_state]))

    for step in range(1, MAX_STEPS + 1):
        # Press ENTER during an episode to end that episode early.
        if btn.enter:
            print('[Episode] Manual early stop.')
            break

        # Keep obstacle avoidance OUT of early RL training.
        # Stop instead, so obstacles do not contaminate line-following learning.
        if ir_sensor.proximity < OBSTACLE_THRESHOLD:
            print('[Episode] Obstacle detected - ending episode.')
            break

        action = select_action(current_state, epsilon)
        execute_action(action)
        sleep(ACTION_TIME)
        stop_motors()
        sleep(0.01)

        next_state = get_current_state()
        reward = get_reward(current_state, action, next_state)
        update_q(current_state, action, reward, next_state)
        total_reward += reward

        if step % 20 == 0:
            print('step {:>3} | refl {:>2} | {:>5} -> {:>7} | r={:>3} | total={}'.format(
                step,
                color_sensor.reflected_light_intensity,
                STATE_NAMES[current_state],
                ACTION_NAMES[action],
                reward,
                total_reward))

        current_state = next_state

    stop_motors()
    return total_reward


def main():
    print('=========================================')
    print(' EV3 Episode-Based Q-Learning Training')
    print('=========================================')
    print('Thresholds: black < {}, edge {}..{}, white >= {}'.format(
        BLACK_MAX, BLACK_MAX, WHITE_MIN - 1, WHITE_MIN))

    if load_q_table():
        print('Existing Q-table loaded.')
        print('Delete {} before a completely fresh training run.'.format(Q_FILE))
    else:
        print('Starting with a fresh Q-table.')

    epsilon = START_EPSILON

    for episode in range(1, EPISODES + 1):
        print('\nSTOPPED: place robot at the SAME starting edge and direction.')
        print('Current reflection:', color_sensor.reflected_light_intensity,
              '| state:', STATE_NAMES[get_current_state()])
        wait_for_enter('Press EV3 ENTER to start episode {}.'.format(episode))

        total_reward = run_episode(episode, epsilon)
        save_q_table()

        print('Episode {} finished | total reward = {}'.format(episode, total_reward))
        print_q_table()

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        sleep(0.3)

    stop_motors()
    save_q_table()
    print('\nTraining complete. Q-table saved to', Q_FILE)
    print_q_table()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as exc:
        print('\nUnexpected error:', exc)
    finally:
        stop_motors()
        try:
            save_q_table()
        except Exception:
            pass
