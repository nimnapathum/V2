#!/usr/bin/env python3

# Import the necessary ev3dev2 classes
import json
import os
import random
from time import sleep

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.button import Button

# ==========================================
# 1. Hardware Initialization
# ==========================================

# Initialize the motors (Assuming left motor is on Port C, right on Port B)
left_motor = LargeMotor(OUTPUT_D)
right_motor = LargeMotor(OUTPUT_A)

# Initialize the sensors
# Color sensor facing down on Port 1, Infrared facing forward on Port 3
color_sensor = ColorSensor(INPUT_1)
ir_sensor = InfraredSensor(INPUT_4)
btn = Button()

# Set the color sensor to measure reflected light intensity (0 to 100)
color_sensor.mode = 'COL-REFLECT'

# Set default motor speeds for smooth control and sharp corner turns
BASE_SPEED = 16
ACTION_TIME = 0.06
BLACK_MAX = 6
WHITE_MIN = 40
OBSTACLE_THRESHOLD = 22
SEARCH_TIMEOUT = 6.0
# Inner-wheel speed during a turn. At -BASE_SPEED the wheels counter-rotate, so a
# turn is pure rotation with no forward creep - the robot holds its ground and
# sweeps the sensor until it finds the edge. This is what lets a 90-degree corner
# be taken by repeatedly applying the same turn action.
# Raise toward -8 to add slow forward creep during turns if it ever stalls.
TURN_SPEED = BASE_SPEED
Q_FILE = "edge_q_table.json"

# ==========================================
# 2. Sensor Reading Functions (For States)
# ==========================================

def get_current_state():
    """
    Reads the color sensor and categorizes it into a discrete state for Edge-Following Q-learning
    (White Line on Black Background):
    - State 0: Outside Line / Lost on Black Mat (Reflected light < 8)
    - State 1: On Edge of White Line / Target Zone (8 <= Reflected light <= 25)
    - State 2: Inside White Line (Reflected light > 25)
    """
    reflection = color_sensor.reflected_light_intensity
    
    if reflection < BLACK_MAX:
        return 0  # State 0: Outside Line (Black Background)
    elif reflection < WHITE_MIN:
        return 1  # State 1: On Edge of White Line (Target Zone)
    else:
        return 2  # State 2: Inside White Line (White Tape)

def check_for_obstacles():
    """
    Reads the infrared sensor proximity (0 to 100).
    A value of ~20 corresponds to approximately 14 cm distance.
    """
    return ir_sensor.proximity

# ==========================================
# 3. Motor Control Functions (For Actions)
# ==========================================

def execute_action(action_id):
    """
    Executes a movement based on the chosen Q-learning action:
    0: Forward (Straight)
    1: Left Pivot Turn (Steers left towards edge when inside white line)
    2: Right Pivot Turn (Steers right towards edge when outside on black mat)
    3: Reverse
    """
    if action_id == 0:
        # Forward
        left_motor.on(SpeedPercent(BASE_SPEED))
        right_motor.on(SpeedPercent(BASE_SPEED))
        
    elif action_id == 1:
        # Left Pivot Turn (counter-rotation: rotates in place, no forward creep)
        left_motor.on(SpeedPercent(-TURN_SPEED))
        right_motor.on(SpeedPercent(BASE_SPEED))

    elif action_id == 2:
        # Right Pivot Turn (counter-rotation: rotates in place, no forward creep)
        left_motor.on(SpeedPercent(BASE_SPEED))
        right_motor.on(SpeedPercent(-TURN_SPEED))
        
    elif action_id == 3:
        # Reverse
        left_motor.on(SpeedPercent(-20))
        right_motor.on(SpeedPercent(-20))

def stop_motors():
    """Halts both motors."""
    left_motor.off()
    right_motor.off()

# ==========================================
# 4. Non-RL Obstacle Avoidance & Path Finding
# ==========================================

def avoid_obstacle_and_find_path():
    """
    Non-RL routine for obstacle avoidance and edge re-acquisition:
    1. Stop motors.
    2. Turn 90 degrees right to bypass obstacle.
    3. Move forward past the obstacle.
    4. Turn 90 degrees left.
    5. Move forward alongside obstacle.
    6. Turn 90 degrees left towards line.
    7. FIND THE EDGE (non-RL): Drive forward scanning until path edge (8-25 reflection) is detected.
    """
    print("\n[OBSTACLE AVOIDANCE] Obstacle detected! Starting non-RL bypass maneuver...")
    stop_motors()
    sleep(0.5)
    
    # 1. Turn Right 90 degrees
    left_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 1.0, block=False)
    right_motor.on_for_seconds(SpeedPercent(-BASE_SPEED), 1.0, block=True)
    
    # 2. Drive forward past obstacle front
    left_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 1.5, block=False)
    right_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 1.5, block=True)
    
    # 3. Turn Left 90 degrees
    left_motor.on_for_seconds(SpeedPercent(-BASE_SPEED), 1.0, block=False)
    right_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 1.0, block=True)
    
    # 4. Drive forward past obstacle side
    left_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 2.0, block=False)
    right_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 2.0, block=True)
    
    # 5. Turn Left 90 degrees towards the line
    left_motor.on_for_seconds(SpeedPercent(-BASE_SPEED), 1.0, block=False)
    right_motor.on_for_seconds(SpeedPercent(BASE_SPEED), 1.0, block=True)
    
    # 6. PATH FINDING (non-RL)
    print("[PATH FINDING] Searching for path edge...")
    left_motor.on(SpeedPercent(15))
    right_motor.on(SpeedPercent(15))
    
    # Search is bounded so the robot cannot drive away forever if the line is missed.
    from time import time
    deadline = time() + SEARCH_TIMEOUT
    found = False
    while time() < deadline:
        reflection = color_sensor.reflected_light_intensity
        if BLACK_MAX <= reflection < WHITE_MIN:
            found = True
            break
        sleep(0.05)

    stop_motors()
    if found:
        print("[PATH FINDING] Path edge re-acquired! Resuming RL control.\n")
    else:
        print("[PATH FINDING] Timed out. Reposition robot near the line.\n")
    sleep(0.5)
    return found

# ==========================================
# 5. Q-Learning Initialization & Rewards
# ==========================================

# 3 States: 0 (Outside Black), 1 (On Edge), 2 (Inside White)
# 4 Actions: 0 (Forward), 1 (Left), 2 (Right), 3 (Reverse)
NUM_STATES = 3
NUM_ACTIONS = 4

ACTION_NAMES = {0: "Forward", 1: "Left Turn", 2: "Right Turn", 3: "Reverse"}
STATE_NAMES = {0: "Outside (Black)", 1: "On Edge (Target)", 2: "Inside (White Line)"}

# Initialize Q-table with zeros
Q_table = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

# Hyperparameters
alpha = 0.45    # Learning rate
gamma = 0.9     # Discount factor
epsilon = 0.30  # Exploration rate 0.12 to learn, 0 to exploit


def get_reward(prev_state, action, next_state):
    """
    Edge-Following Reward logic (White Line on Black Background):
    - State 1 (On Edge): Target zone! Forward (+10) is strongly favored. A turn that
      RECOVERS the edge from off-line scores +8, but a turn taken while ALREADY on
      the edge scores only +2 - otherwise wobbling in place pays nearly as well as
      driving straight, and the policy learns to pivot forever instead of advancing.
    - State 0 (Outside on Black Mat): Continuing forward (-10). Turning Right (+7).
    - State 2 (Inside White Line): Turning Left (Action 1) towards edge (+7).
    - Leaving the edge is penalised by INTENT: drifting off while going straight is
      a mild -3, but steering off deliberately is -8. The old code did the reverse
      (-10 for drifting off, -2 for steering off), which taxed the one action we
      most want and made turning on the edge the profitable choice.
    """
    # Target state: Reached or stayed on edge (State 1)
    if next_state == 1:
        if action == 0:
            return 10    # Max reward for going straight along the edge
        elif action in (1, 2):
            # Recovering the edge is valuable; wobbling on it is not.
            return 2 if prev_state == 1 else 8
        else:
            return -5    # Reverse on edge

    # Outside on black mat (State 0)
    elif next_state == 0:
        if prev_state == 1:
            return -3 if action == 0 else -8   # drifted off vs steered off
        elif action == 0:
            return -10   # Heavy penalty for continuing forward when lost on black mat!
        elif action == 2:
            return 7     # Turning Right from black mat towards line edge
        else:
            return -6

    # Inside white line (State 2)
    else:
        if prev_state == 1:
            return -3 if action == 0 else -8   # drifted off vs steered off
        elif action == 1:
            return 7     # Turning Left from white line towards edge
        else:
            return -3    # Driving deeper into the tape is never useful

def select_action(state):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)

    # Random tie-breaking prevents the initial all-zero table from always
    # selecting action 0 (Forward).
    max_val = max(Q_table[state])
    best = [a for a, value in enumerate(Q_table[state]) if value == max_val]
    return random.choice(best)

def save_q_table():
    """Saves Q-table to JSON file for persistent trained performance."""
    try:
        with open(Q_FILE, 'w') as f:
            json.dump(Q_table, f)
        print("\nQ-table saved to {}".format(Q_FILE))
    except Exception as e:
        print("Failed to save Q-table: {}".format(e))

def load_q_table():
    """Loads pre-trained Q-table if available."""
    global Q_table
    if os.path.exists(Q_FILE):
        try:
            with open(Q_FILE, 'r') as f:
                loaded = json.load(f)
                if (not isinstance(loaded, list) or len(loaded) != NUM_STATES or
                        any(not isinstance(row, list) or len(row) != NUM_ACTIONS for row in loaded)):
                    raise ValueError('Q-table dimensions do not match current state/action space')
                Q_table = [[float(v) for v in row] for row in loaded]
            print("Loaded pre-trained Q-table from {}".format(Q_FILE))
        except Exception as e:
            print("Could not load Q-table: {}".format(e))

# ==========================================
# 6. Main Loop Execution
# ==========================================

if __name__ == '__main__':
    running_flag = True

    try:
        print("==========================================")
        print("   EV3 Q-Learning Edge Follower Starting  ")
        print("==========================================")
        
        load_q_table()
        current_state = get_current_state()
        
        step_count = 0
        while running_flag:
            # 0. Check for manual EV3 Middle Button (Enter) press to stop
            if btn.enter:
                print("\n[BUTTON] Middle button pressed. Gracefully stopping...")
                running_flag = False
                break
                
            # 1. Non-RL Obstacle Avoidance Override (IR Proximity < 20)
            if check_for_obstacles() < OBSTACLE_THRESHOLD:
                avoid_obstacle_and_find_path()
                current_state = get_current_state()
                continue
            
            # 2. RL Decision Pipeline
            chosen_action = select_action(current_state)
            execute_action(chosen_action)
            
            # Action execution duration. At 0.12s a single turn swept the sensor
            # ~1.2cm - wider than the tape, so corrections overshot the edge and
            # the robot hunted. 0.05s gives ~0.5cm per step: sub-tape resolution.
            sleep(ACTION_TIME)
            stop_motors()
            sleep(0.01)
            
            # Observe environment transition
            next_state = get_current_state()
            reward = get_reward(current_state, chosen_action, next_state)
            
            # Q-learning Bellman Update Equation: 
            # Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max(Q(s', a')) - Q(s, a)]
            best_next_action_val = max(Q_table[next_state])
            Q_table[current_state][chosen_action] += alpha * (
                reward + gamma * best_next_action_val - Q_table[current_state][chosen_action]
            )
            
            step_count += 1
            if step_count % 5 == 0:
                refl = color_sensor.reflected_light_intensity
                print("Refl: {:<2} | State: {:<20} | Action: {:<10} | Reward: {:<3}".format(
                    refl,
                    STATE_NAMES[current_state],
                    ACTION_NAMES[chosen_action],
                    reward
                ))
            
            current_state = next_state
            
    except Exception as e:
        print("Unexpected error: {}".format(e))
    finally:
        print("\nStopping robot...")
        stop_motors()
        save_q_table()
        
        print("\nFinal Learned Q-Table:")
        print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("State", "Forward", "Left", "Right", "Reverse"))
        for s in range(NUM_STATES):
            print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
                STATE_NAMES[s],
                Q_table[s][0],
                Q_table[s][1],
                Q_table[s][2],
                Q_table[s][3]
            ))
            
        print("\n[DISPLAY] Keeping training summary visible for 10 seconds before closing...")
        sleep(10)
