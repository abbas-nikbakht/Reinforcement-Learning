# Simulating the GridWorld 3×4 Environment and Human Play
> [!IMPORTANT]
> This environment is implemented from scratch without using Gymnasium.
In this project, a **3×4 GridWorld environment is implemented from scratch without using Gymnasium**.

The player can interact with the environment using the keyboard arrow keys (up, down, left, and right) and try to reach the goal, represented by the green cell.

---

## 📌 Overview

This project provides an interactive **3×4 GridWorld environment implemented from scratch using Python and NumPy**.

<span style="color:red"> **Unlike environments provided by libraries such as Gymnasium, the core environment dynamics are implemented manually.**</span> This includes the states, actions, transitions, rewards, terminal states, and stochastic behavior of the environment.

A human player can interact with the environment using keyboard arrow keys and observe how actions affect the agent's state and rewards.

<span style="color:red"> Objective:</span> The main purpose of this project is to understand how a Reinforcement Learning environment works internally before using pre-built environments or implementing Reinforcement Learning algorithms.

---

## Features

* 🎮 Interactive GridWorld gameplay using keyboard arrow keys
* 🗺️ 3×4 grid-based environment
* 🤖 Manual control of the agent
* 🎯 Goal and terminal states
* 🔄 Stochastic state transitions
* 💰 Reward-based interaction
* 📍 Visualization of states and agent movements
* 🧠 Environment implemented from scratch
* 🔢 Transition probabilities implemented using NumPy
* 🆕 New Episode functionality
* **🚫 No Gymnasium dependency**

---

## Technologies

| Technology | Purpose                                         |
| ---------- | ----------------------------------------------- |
| 🐍 Python  | Core programming language                       |
| 🔢 NumPy   | Environment dynamics and stochastic transitions |
| 🕹️ Pygame | Interactive visualization and keyboard input    |

---

## Installation

Install the required dependencies:

```bash
pip install numpy pygame
```

---

## Run the Application

```bash
python gridworld_3x4_from_scratch.py
```

Use the keyboard arrow keys to control the agent:

⬆️ Move Up
⬇️ Move Down
⬅️ Move Left
➡️ Move Right

---

## 🧠 Concepts Demonstrated

This project provides a practical introduction to several important Reinforcement Learning concepts:

* **GridWorld Environment**
* **Agent–Environment Interaction**
* **States and Actions**
* **Rewards and Terminal States**
* **State Transitions**
* **Transition Probabilities**
* **Stochastic Environments**
* **Markov Decision Process (MDP)**
* **Environment Dynamics**
* **Reinforcement Learning Fundamentals**
* **Environment Visualization**

---

## 🚀 Future Improvements

Possible extensions of this project include:

* [x] Interactive environment
* [x] Keyboard-controlled agent
* [x] Graphical environment
* Display rewards and game information
* [x] Environment implemented from scratch
* [ ] Improve the graphical interface
* [ ] Add different GridWorld configurations

---

## 👤 Author

**Abbas Nikbakht**

<span style="color:green"> This project was developed to understand the internal structure and dynamics of environments used in Reinforcement Learning by implementing a GridWorld environment from scratch.</span>
