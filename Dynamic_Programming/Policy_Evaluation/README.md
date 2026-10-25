# GridWorld 3×4 — Iterative Policy Evaluation

This project demonstrates how the **Iterative Policy Evaluation** algorithm can be used to estimate the state-value function of a fixed policy in a stochastic 3×4 GridWorld environment.

The application is built using Python, Gymnasium, Plotly, and Dash. It provides an interactive visualization of the GridWorld and updates the value of each state as the algorithm iteratively evaluates the given policy.

---

## 😊 **Overview**

**Iterative Policy Evaluation** is a Dynamic Programming algorithm used in Reinforcement Learning to estimate the value function of a given policy.

In this project, a 3×4 GridWorld environment is created with a starting state, a goal, a hole, and a wall. The environment uses stochastic transitions, meaning that an action does not always lead to the intended next state.

The state-value function is updated using the Bellman Expectation Equation:

```math
V(s)=\sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]
```

Where:

- (V(s)\) is the estimated value of state \(s\).
- (pi(a|s)\) is the probability of taking action \(a\) in state \(s\).
- (p(s',r|s,a)\) is the probability of transitioning to state \(s'\) and receiving reward \(r\).
- (r\) is the immediate reward.
- (gamma\) is the discount factor.

The algorithm repeatedly updates the state values until the maximum change between successive estimates becomes smaller than a predefined threshold \(\theta\).

The maximum change is calculated as:

```math
\Delta=\max_s|V_{\text{new}}(s)-V_{\text{old}}(s)|
```

The stopping condition is:

```math
\Delta < \theta
```

The convergence threshold can be adjusted through the interactive application.

---

## Technologies

| **Technology** | **Purpose** |
|---|---|
| Python | Core programming language |
| Gymnasium | GridWorld environment and transition dynamics |
| NumPy | Numerical calculations |
| Plotly | Interactive GridWorld visualization |
| Dash | Interactive web application |
| 🎯 Dynamic Programming | Evaluation of a fixed policy |

---

## Installation

Install the required dependencies:

```bash
pip install dash plotly gymnasium numpy
```

---

## Run the Application

Run the Python script:

```bash
python gridworld_iterative_policy_evaluation.py
```

Replace `gridworld_iterative_policy_evaluation.py` with the actual name of your Python file.

After starting the application, open the following local URL in your browser:

[http://127.0.0.1:8051](http://127.0.0.1:8051)

---

## Future Improvements

Possible extensions of this project include:

- ✅ Add an interactive input for the convergence threshold \(\theta\).
- ✅ Display state-value estimates directly inside the GridWorld.
- ✅ Update the visualization during the evaluation process.
- Add an option to select or modify the policy.
- Allow users to adjust the discount factor \(\gamma\).
- Add a reset button to restart the evaluation.
- Improve the visualization and user interface.

---

## Author

**Abbas Nikbakht**

This project was developed to demonstrate **Iterative Policy Evaluation** and understand how Dynamic Programming can be used to estimate state values under a fixed policy in Reinforcement Learning.
