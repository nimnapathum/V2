#!/usr/bin/env python3
"""
LEGO EV3 RL Path Follower - Assignment Version

Features
--------
* Q-learning with 4 learned actions: Forward, Left, Right, Reverse
* 18 states = 2 travel directions (CW/CCW) x 9 previous/current light-zone states
* Outcome-based rewards: no rule says "black -> right" or "white -> left"
* Episode-based physical training with manual reset
* Alternating CW / CCW training episodes
* Smooth differential steering for Left/Right actions
* Persistent model (Q-table + epsilon + completed episode count)
* CSV training statistics
* TEST mode with epsilon=0 and no Q updates
* Deterministic obstacle avoidance and path reacquisition (allowed by assignment)

IMPORTANT BEFORE FIRST RUN
--------------------------
1. Calibrate BLACK_MAX and WHITE_MIN for YOUR track.
2. Verify motor ports and sensor ports.
3. Start with MODE = "TRAIN" and NO obstacles.
4. Delete MODEL_FILE if you want a completely fresh experiment.
"""

import csv
import json
import os
import random
from time import sleep, time

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.button import Button

# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# "TRAIN" = learn in separate episodes
# "TEST"  = load trained model, epsilon=0, no Q updates, continuous control
MODE = "TRAIN"

# In TEST mode choose the direction you are about to demonstrate.
# Use "CW" or "CCW".
TEST_DIRECTION = "CW"

# Number of new physical episodes performed each time TRAIN mode is run.
EPISODES_THIS_RUN = 20
MAX_STEPS_PER_EPISODE = 350

# ---------------- Sensor calibration ----------------
# Replace these after measuring YOUR black / edge / white reflected-light values.
BLACK_MAX = 10
WHITE_MIN = 30

# ---------------- Motor / timing ----------------
FORWARD_SPEED = 18
TURN_OUTER_SPEED = 18
TURN_INNER_SPEED = 5
REVERSE_SPEED = 14

# One RL action lasts this long during TRAIN mode.
TRAIN_ACTION_TIME = 0.08
TRAIN_SETTLE_TIME = 0.015

# In TEST mode motors stay running; the policy is reconsidered at this interval.
TEST_DECISION_TIME = 0.055

# ---------------- Obstacle detection ----------------
OBSTACLE_THRESHOLD = 20

# Encoder values are robot-specific: CALIBRATE these on your EV3.
TURN_90_MOTOR_DEGREES = 330
BYPASS_FORWARD_DEGREES = 650
BYPASS_SIDE_DEGREES = 900
SEARCH_SPEED = 13
SEARCH_TIMEOUT = 7.0

# ---------------- Files ----------------
MODEL_FILE = "ev3_rl_model.json"
STATS_FILE = "ev3_training_stats.csv"

# ============================================================
# 2. HARDWARE
# ============================================================

# Change these if your wiring is different.
left_motor = LargeMotor(OUTPUT_D)
right_motor = LargeMotor(OUTPUT_A)

color_sensor = ColorSensor(INPUT_1)
ir_sensor = InfraredSensor(INPUT_4)
btn = Button()

color_sensor.mode = 'COL-REFLECT'

# ============================================================
# 3. RL DEFINITIONS
# ============================================================

# Light zones
BLACK = 0
EDGE = 1
WHITE = 2
ZONE_NAMES = {
    BLACK: "Black",
    EDGE: "Edge",
    WHITE: "White",
}

# Directions are CONTEXT, not actions.
# The agent still learns which motor action is best in each direction/state.
CW = 0
CCW = 1
DIRECTION_NAMES = {
    CW: "CW",
    CCW: "CCW",
}

# Learned actions required by assignment
FORWARD = 0
LEFT = 1
RIGHT = 2
REVERSE = 3
NUM_ACTIONS = 4
ACTION_NAMES = {
    FORWARD: "Forward",
    LEFT: "Left",
    RIGHT: "Right",
    REVERSE: "Reverse",
}

# 9 transition states per direction:
# (previous_zone, current_zone) = 3 x 3
TRANSITION_STATES = 9
NUM_DIRECTIONS = 2
NUM_STATES = TRANSITION_STATES * NUM_DIRECTIONS  # 18 total

# Q-learning hyperparameters
ALPHA = 0.40
GAMMA = 0.90
START_EPSILON = 0.35
MIN_EPSILON = 0.04
EPSILON_DECAY = 0.90

Q_table = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
current_epsilon = START_EPSILON
completed_episodes = 0

# ============================================================
# 4. STATE / SENSOR FUNCTIONS
# ============================================================

def read_reflection():
    return color_sensor.reflected_light_intensity


def get_zone(reflection=None):
    """Convert reflected-light value into Black / Edge / White zone."""
    if reflection is None:
        reflection = read_reflection()

    if reflection < BLACK_MAX:
        return BLACK
    if reflection < WHITE_MIN:
        return EDGE
    return WHITE


def state_index(direction, previous_zone, current_zone):
    """Encode direction + previous/current zone into one state index [0..17]."""
    transition = previous_zone * 3 + current_zone
    return direction * TRANSITION_STATES + transition


def decode_state(state):
    direction = state // TRANSITION_STATES
    transition = state % TRANSITION_STATES
    previous_zone = transition // 3
    current_zone = transition % 3
    return direction, previous_zone, current_zone


def state_name(state):
    direction, previous_zone, current_zone = decode_state(state)
    return "{}:{}->{}".format(
        DIRECTION_NAMES[direction],
        ZONE_NAMES[previous_zone],
        ZONE_NAMES[current_zone],
    )

# ============================================================
# 5. MOTOR ACTIONS
# ============================================================

def execute_action(action):
    """
    Define what each action physically means.

    IMPORTANT: This does NOT decide WHEN to use an action. Q-learning does that.
    Left/Right use differential steering instead of in-place pivots for smoothness.
    """
    if action == FORWARD:
        left_motor.on(SpeedPercent(FORWARD_SPEED))
        right_motor.on(SpeedPercent(FORWARD_SPEED))

    elif action == LEFT:
        left_motor.on(SpeedPercent(TURN_INNER_SPEED))
        right_motor.on(SpeedPercent(TURN_OUTER_SPEED))

    elif action == RIGHT:
        left_motor.on(SpeedPercent(TURN_OUTER_SPEED))
        right_motor.on(SpeedPercent(TURN_INNER_SPEED))

    elif action == REVERSE:
        left_motor.on(SpeedPercent(-REVERSE_SPEED))
        right_motor.on(SpeedPercent(-REVERSE_SPEED))


def stop_motors(brake=True):
    left_motor.off(brake=brake)
    right_motor.off(brake=brake)

# ============================================================
# 6. OUTCOME-BASED REWARD
# ============================================================

def get_reward(previous_zone, current_zone, next_zone):
    """
    Reward WHAT HAPPENED, not WHICH NAMED ACTION was selected.

    There are deliberately no checks such as:
        if action == LEFT: reward = ...
        if action == RIGHT: reward = ...

    This lets Q-learning discover which action produces good outcomes.

    Desired outcome: remain on or recover the edge.
    """

    # Best outcome: edge is currently maintained.
    if current_zone == EDGE and next_zone == EDGE:
        return 12

    # Strong recovery: robot was away from edge and reacquired it.
    if current_zone != EDGE and next_zone == EDGE:
        return 15

    # Leaving a good edge position is bad.
    if current_zone == EDGE and next_zone != EDGE:
        return -8

    # Still stuck on the same wrong side.
    if current_zone == next_zone and current_zone != EDGE:
        return -6

    # Crossed directly from one side to the other without sampling EDGE.
    # Usually indicates an overshoot / too aggressive correction.
    if ((current_zone == BLACK and next_zone == WHITE) or
            (current_zone == WHITE and next_zone == BLACK)):
        return -5

    # Changing while off the edge may represent an attempted recovery,
    # but until EDGE is actually reached it should not receive a big reward.
    return -2

# ============================================================
# 7. Q-LEARNING
# ============================================================

def select_action(state, epsilon):
    """Epsilon-greedy selection with random tie breaking."""
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)

    best_value = max(Q_table[state])
    best_actions = [
        action for action, value in enumerate(Q_table[state])
        if value == best_value
    ]
    return random.choice(best_actions)


def greedy_action(state):
    """Pure exploitation for final TEST mode."""
    return select_action(state, 0.0)


def update_q(state, action, reward, next_state):
    old_q = Q_table[state][action]
    best_future_q = max(Q_table[next_state])

    Q_table[state][action] = old_q + ALPHA * (
        reward + GAMMA * best_future_q - old_q
    )

# ============================================================
# 8. MODEL PERSISTENCE
# ============================================================

def save_model():
    payload = {
        "version": 2,
        "num_states": NUM_STATES,
        "num_actions": NUM_ACTIONS,
        "q_table": Q_table,
        "epsilon": current_epsilon,
        "completed_episodes": completed_episodes,
        "black_max": BLACK_MAX,
        "white_min": WHITE_MIN,
    }

    with open(MODEL_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def load_model():
    global Q_table, current_epsilon, completed_episodes

    if not os.path.exists(MODEL_FILE):
        return False

    try:
        with open(MODEL_FILE, "r") as f:
            payload = json.load(f)

        loaded_q = payload["q_table"]

        if len(loaded_q) != NUM_STATES:
            raise ValueError("Saved Q-table has wrong number of states")

        if any(len(row) != NUM_ACTIONS for row in loaded_q):
            raise ValueError("Saved Q-table has wrong number of actions")

        Q_table = [[float(value) for value in row] for row in loaded_q]
        current_epsilon = float(payload.get("epsilon", START_EPSILON))
        completed_episodes = int(payload.get("completed_episodes", 0))

        return True

    except Exception as exc:
        print("Could not load model: {}".format(exc))
        return False

# ============================================================
# 9. TRAINING STATISTICS
# ============================================================

def append_stats(stats):
    exists = os.path.exists(STATS_FILE)

    fieldnames = [
        "episode",
        "direction",
        "epsilon",
        "steps",
        "total_reward",
        "edge_steps",
        "edge_percent",
        "forward_count",
        "left_count",
        "right_count",
        "reverse_count",
        "manual_stop",
    ]

    with open(STATS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(stats)

# ============================================================
# 10. BUTTON / DISPLAY HELPERS
# ============================================================

def wait_for_enter(message):
    print(message)

    # Ensure a previous press is released.
    while btn.enter:
        sleep(0.05)

    while not btn.enter:
        sleep(0.05)

    while btn.enter:
        sleep(0.05)


def print_q_table(direction=None):
    print("\n{:<20} {:>9} {:>9} {:>9} {:>9}".format(
        "State", "Forward", "Left", "Right", "Reverse"
    ))

    for state in range(NUM_STATES):
        state_direction, _, _ = decode_state(state)
        if direction is not None and state_direction != direction:
            continue

        print("{:<20} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}".format(
            state_name(state),
            Q_table[state][FORWARD],
            Q_table[state][LEFT],
            Q_table[state][RIGHT],
            Q_table[state][REVERSE],
        ))

# ============================================================
# 11. NON-RL OBSTACLE AVOIDANCE
# ============================================================

def turn_right_90():
    """Deterministic encoder-based turn. CALIBRATE TURN_90_MOTOR_DEGREES."""
    left_motor.on_for_degrees(
        SpeedPercent(FORWARD_SPEED),
        TURN_90_MOTOR_DEGREES,
        brake=True,
        block=False,
    )
    right_motor.on_for_degrees(
        SpeedPercent(-FORWARD_SPEED),
        TURN_90_MOTOR_DEGREES,
        brake=True,
        block=True,
    )


def turn_left_90():
    left_motor.on_for_degrees(
        SpeedPercent(-FORWARD_SPEED),
        TURN_90_MOTOR_DEGREES,
        brake=True,
        block=False,
    )
    right_motor.on_for_degrees(
        SpeedPercent(FORWARD_SPEED),
        TURN_90_MOTOR_DEGREES,
        brake=True,
        block=True,
    )


def drive_forward_degrees(degrees, speed=FORWARD_SPEED):
    left_motor.on_for_degrees(
        SpeedPercent(speed), degrees, brake=True, block=False
    )
    right_motor.on_for_degrees(
        SpeedPercent(speed), degrees, brake=True, block=True
    )


def find_path_non_rl():
    """
    Search forward for the line edge after the obstacle bypass.
    This is deliberately NON-RL because the assignment allows it.
    """
    print("[PATH FINDING] Searching for edge...")

    deadline = time() + SEARCH_TIMEOUT
    left_motor.on(SpeedPercent(SEARCH_SPEED))
    right_motor.on(SpeedPercent(SEARCH_SPEED))

    found = False

    while time() < deadline:
        zone = get_zone()
        if zone == EDGE:
            found = True
            break
        sleep(0.03)

    stop_motors()

    if found:
        print("[PATH FINDING] Edge reacquired.")
        sleep(0.15)
        return True

    print("[PATH FINDING] Timed out. Stop and reposition robot safely.")
    return False


def avoid_obstacle_non_rl():
    """
    Simple rectangular bypass around an obstacle.

    These motions are NOT learned; that is allowed by the assignment.
    Tune motor degrees to your physical obstacle size and robot geometry.
    """
    print("\n[OBSTACLE] Detected. Pausing RL policy.")
    stop_motors()
    sleep(0.2)

    # Step around the obstacle.
    turn_right_90()
    drive_forward_degrees(BYPASS_FORWARD_DEGREES)
    turn_left_90()
    drive_forward_degrees(BYPASS_SIDE_DEGREES)
    turn_left_90()

    # Move toward the expected path and detect it with the color sensor.
    found = find_path_non_rl()

    if not found:
        stop_motors()
        return False

    print("[OBSTACLE] Path recovered. Returning to RL policy.\n")
    return True

# ============================================================
# 12. TRAINING EPISODE
# ============================================================

def run_training_episode(global_episode, direction, epsilon):
    """Run one bounded physical training episode."""
    print("\n================================================")
    print("EPISODE {} | {} | epsilon={:.3f}".format(
        global_episode, DIRECTION_NAMES[direction], epsilon
    ))
    print("================================================")

    total_reward = 0
    steps = 0
    edge_steps = 0
    manual_stop = False

    action_counts = {
        FORWARD: 0,
        LEFT: 0,
        RIGHT: 0,
        REVERSE: 0,
    }

    # Initialize transition history from the current physical position.
    current_zone = get_zone()
    previous_zone = current_zone
    state = state_index(direction, previous_zone, current_zone)

    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        # ENTER is an emergency / early episode stop.
        if btn.enter:
            manual_stop = True
            print("[TRAIN] Manual early stop.")
            while btn.enter:
                sleep(0.03)
            break

        # Obstacles are excluded from RL training so they do not contaminate
        # line-following learning. The final TEST mode handles them separately.
        if ir_sensor.proximity < OBSTACLE_THRESHOLD:
            print("[TRAIN] Obstacle detected. Ending this training episode.")
            break

        action = select_action(state, epsilon)
        action_counts[action] += 1

        execute_action(action)
        sleep(TRAIN_ACTION_TIME)
        stop_motors()
        sleep(TRAIN_SETTLE_TIME)

        next_zone = get_zone()

        # IMPORTANT: reward uses environmental outcome only.
        reward = get_reward(previous_zone, current_zone, next_zone)

        next_state = state_index(direction, current_zone, next_zone)
        update_q(state, action, reward, next_state)

        total_reward += reward
        steps += 1

        if next_zone == EDGE:
            edge_steps += 1

        if step % 25 == 0:
            print(
                "step {:>3} | refl {:>2} | {:<17} | action {:<7} | r {:>3} | total {:>5}".format(
                    step,
                    read_reflection(),
                    state_name(state),
                    ACTION_NAMES[action],
                    reward,
                    total_reward,
                )
            )

        previous_zone = current_zone
        current_zone = next_zone
        state = next_state

    stop_motors()

    edge_percent = (100.0 * edge_steps / steps) if steps else 0.0

    stats = {
        "episode": global_episode,
        "direction": DIRECTION_NAMES[direction],
        "epsilon": round(epsilon, 4),
        "steps": steps,
        "total_reward": total_reward,
        "edge_steps": edge_steps,
        "edge_percent": round(edge_percent, 2),
        "forward_count": action_counts[FORWARD],
        "left_count": action_counts[LEFT],
        "right_count": action_counts[RIGHT],
        "reverse_count": action_counts[REVERSE],
        "manual_stop": manual_stop,
    }

    return stats

# ============================================================
# 13. TRAIN MODE
# ============================================================

def training_main():
    global current_epsilon, completed_episodes

    print("============================================")
    print(" EV3 Q-LEARNING - TRAIN MODE")
    print("============================================")
    print("Black < {} | Edge {}..{} | White >= {}".format(
        BLACK_MAX, BLACK_MAX, WHITE_MIN - 1, WHITE_MIN
    ))

    if load_model():
        print("Loaded trained model: {}".format(MODEL_FILE))
        print("Completed episodes: {}".format(completed_episodes))
        print("Resuming epsilon: {:.3f}".format(current_epsilon))
    else:
        print("No existing model found. Starting fresh.")
        current_epsilon = START_EPSILON
        completed_episodes = 0

    for local_episode in range(1, EPISODES_THIS_RUN + 1):
        global_episode = completed_episodes + 1

        # Alternate CW / CCW so both policies receive training.
        direction = CW if global_episode % 2 == 1 else CCW

        print("\n------------------------------------------------")
        print("NEXT: Episode {} ({})".format(
            global_episode, DIRECTION_NAMES[direction]
        ))
        print("1. Pick up / reset robot to the training start.")
        print("2. Point it in the {} travel direction.".format(
            DIRECTION_NAMES[direction]
        ))
        print("3. Put color sensor approximately over the EDGE.")
        print("4. Keep obstacles OUT of the RL training course.")
        print("Current reflection={} | zone={}".format(
            read_reflection(), ZONE_NAMES[get_zone()]
        ))

        wait_for_enter("Press EV3 ENTER to start this episode.")

        stats = run_training_episode(
            global_episode,
            direction,
            current_epsilon,
        )

        append_stats(stats)

        # Episode is now complete.
        completed_episodes = global_episode
        current_epsilon = max(
            MIN_EPSILON,
            current_epsilon * EPSILON_DECAY,
        )

        save_model()

        print("\nEpisode {} finished".format(global_episode))
        print("Direction       : {}".format(stats["direction"]))
        print("Steps           : {}".format(stats["steps"]))
        print("Total reward    : {}".format(stats["total_reward"]))
        print("Edge percentage : {:.2f}%".format(stats["edge_percent"]))
        print("Actions         : F={} L={} R={} Rev={}".format(
            stats["forward_count"],
            stats["left_count"],
            stats["right_count"],
            stats["reverse_count"],
        ))
        print("Next epsilon    : {:.3f}".format(current_epsilon))

        print_q_table(direction)
        sleep(0.25)

    print("\n============================================")
    print("TRAINING RUN COMPLETE")
    print("Model saved to : {}".format(MODEL_FILE))
    print("Stats saved to : {}".format(STATS_FILE))
    print("Completed eps   : {}".format(completed_episodes))
    print("Current epsilon : {:.3f}".format(current_epsilon))
    print("============================================")

# ============================================================
# 14. TEST MODE
# ============================================================

def test_main():
    """
    Final demonstration mode:
    - loads learned Q-table
    - epsilon = 0
    - DOES NOT update Q-values
    - motors remain continuous for smoother motion
    - obstacle avoidance and path finding are enabled
    """
    if not load_model():
        print("ERROR: No trained model found. Train the robot first.")
        return

    direction_text = TEST_DIRECTION.strip().upper()
    if direction_text == "CW":
        direction = CW
    elif direction_text == "CCW":
        direction = CCW
    else:
        raise ValueError('TEST_DIRECTION must be "CW" or "CCW"')

    print("============================================")
    print(" EV3 Q-LEARNING - TEST MODE")
    print("============================================")
    print("Direction : {}".format(DIRECTION_NAMES[direction]))
    print("epsilon   : 0.0 (NO random exploration)")
    print("Q updates : OFF")
    print("Obstacle avoidance : ON")
    print("Press ENTER during test to stop.")

    print_q_table(direction)

    wait_for_enter(
        "Place robot on edge pointing {} and press ENTER to start TEST.".format(
            DIRECTION_NAMES[direction]
        )
    )

    current_zone = get_zone()
    previous_zone = current_zone

    while True:
        if btn.enter:
            print("[TEST] Stop requested.")
            while btn.enter:
                sleep(0.03)
            break

        if ir_sensor.proximity < OBSTACLE_THRESHOLD:
            stop_motors()

            if not avoid_obstacle_non_rl():
                print("[TEST] Could not reacquire path. Test stopped.")
                break

            # Reset temporal state after a deterministic obstacle maneuver.
            current_zone = get_zone()
            previous_zone = current_zone
            continue

        state = state_index(direction, previous_zone, current_zone)
        action = greedy_action(state)

        # Do NOT stop motors after each action in TEST mode.
        # New speed commands flow into each other for smoother movement.
        execute_action(action)
        sleep(TEST_DECISION_TIME)

        next_zone = get_zone()
        previous_zone = current_zone
        current_zone = next_zone

    stop_motors()
    print("TEST finished.")

# ============================================================
# 15. ENTRY POINT
# ============================================================

def main():
    mode = MODE.strip().upper()

    if mode == "TRAIN":
        training_main()
    elif mode == "TEST":
        test_main()
    else:
        raise ValueError('MODE must be "TRAIN" or "TEST"')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by keyboard.")
    except Exception as exc:
        print("\nUnexpected error: {}".format(exc))
    finally:
        stop_motors()
        # Save only when training; TEST should not alter the trained model.
        if MODE.strip().upper() == "TRAIN":
            try:
                save_model()
            except Exception:
                pass
