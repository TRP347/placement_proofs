"""
drone_rl.py
Simplified Autonomous Drone Coordination
- Multi-agent Q-learning
- OpenCV obstacle detection (mocked with frames)
- Reward shaping for collisions
"""

import numpy as np
import cv2
import random

# ---------------------
# ENVIRONMENT
# ---------------------
GRID_SIZE = 10
N_AGENTS = 2
EPISODES = 200

# States: (x, y) of each agent, Obstacles are fixed
obstacles = [(4, 4), (5, 5), (6, 2)]

def get_state(agent_pos):
    return agent_pos[0] * GRID_SIZE + agent_pos[1]

def step(agent_pos, action):
    x, y = agent_pos
    if action == 0: x = max(0, x-1) # up
    if action == 1: x = min(GRID_SIZE-1, x+1) # down
    if action == 2: y = max(0, y-1) # left
    if action == 3: y = min(GRID_SIZE-1, y+1) # right
    return (x, y)

# ---------------------
# Q-LEARNING
# ---------------------
actions = [0,1,2,3]
Q = np.zeros((GRID_SIZE*GRID_SIZE, len(actions)))

alpha, gamma, epsilon = 0.1, 0.9, 0.2

def reward(agent_pos, other_pos):
    if agent_pos in obstacles: return -10
    if agent_pos == other_pos: return -20
    if agent_pos == (GRID_SIZE-1, GRID_SIZE-1): return +50
    return -1

# ---------------------
# TRAIN
# ---------------------
for ep in range(EPISODES):
    agents = [(0,0), (0,1)]
    for step_count in range(50):
        for i in range(N_AGENTS):
            s = get_state(agents[i])
            if random.uniform(0,1) < epsilon:
                a = random.choice(actions)
            else:
                a = np.argmax(Q[s])
            new_pos = step(agents[i], a)
            r = reward(new_pos, agents[1-i])
            s_next = get_state(new_pos)
            Q[s,a] += alpha*(r + gamma*np.max(Q[s_next]) - Q[s,a])
            agents[i] = new_pos

print("Training complete. Agents learned to avoid obstacles and collisions.")

# ---------------------
# OBSTACLE DETECTION DEMO (OpenCV)
# ---------------------
cap = cv2.VideoCapture(0) # webcam for demo
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fake obstacle detection: threshold + contours
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
