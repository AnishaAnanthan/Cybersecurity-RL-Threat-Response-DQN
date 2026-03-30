# Cybersecurity Threat Response using Deep Reinforcement Learning (DQN)

## Overview
This project implements a Reinforcement Learning-based cybersecurity system that analyzes network traffic and predicts appropriate actions such as:
- Allow Traffic
- Block IP
- Monitor Activity
- Alert Admin

The system uses a Deep Q-Network (DQN) to learn optimal decisions based on reward mechanisms.

---

## Objectives
- Automate cybersecurity decision-making
- Reduce manual monitoring effort
- Provide explainable AI outputs
- Improve response accuracy using RL

---

## Algorithm Used
**Deep Q-Network (DQN)**
- Uses neural networks to estimate Q-values
- Selects actions based on maximum expected reward
- Learns optimal policy through experience

---

## How It Works
1. Input network traffic features
2. Normalize values (0 to 1)
3. RL agent predicts best action
4. Reward system evaluates correctness
5. Outputs:
   - Recommended action
   - Confidence scores
   - Explanation

---

## Input Features
The model uses 10 important features such as:
- Flow Duration
- Packet Length
- Flow Bytes/s
- Packet Statistics

(All features are normalized between 0 and 1)

---

## Actions
| Action | Meaning |
|------|--------|
| 0 | Allow Traffic |
| 1 | Block IP |
| 2 | Monitor |
| 3 | Alert |

---

## Reward Logic
- High reward → Correct action
- Medium reward → Acceptable action
- Negative reward → Incorrect decision
- Strong penalty for unsafe actions

---

## Tech Stack
- Python
- NumPy, Pandas
- Gymnasium
- Stable-Baselines3
- PyTorch
- Streamlit

---

## 📁 Project Structure
