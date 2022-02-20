# Kinodynamic-RRT-Path-Planning
## Introduction
The term “Kinodynamic planning” was first introduced by Donald et al. [1] in 1993 and has been a research hotspot since then. Kinodynamic planning problems simultaneously synthesize a robot motion subject to both kinematic and dynamics constraints. Specifically, kinematic constraints, such as avoiding obstacles, limit the configuration of a robot. Dynamics constraints, such as bounds on velocity and acceleration, govern the time-derivatives of the configuration. The state of a robot in this planning problem is typically defined as its configuration and velocity. Kinodynamic planning problems can be formulated as making a robot find a motion that goes from the initial state to the goal state while obeying kinodynamic constraints.
This project aims to implement the kinodynamic planning based on the RRT-connect algorithm with fixed time step and best-input extension for a planar hover-craft robot. The robot can accelerate or decelerate in x, y, and θ while moving from an initial state to a goal state.

## Report
https://drive.google.com/file/d/1DxIJ-e8PBasTWaxDPd7EP_qJIPRqXAcb/view?usp=sharing

## Video Demo
https://www.youtube.com/watch?v=66A5YRcBMao

## References
[1]	B. Donald, P. Xavier, J. Canny, and J. Reif, “Kinodynamic motion planning,” J. ACM, vol. 40, no. 5, pp. 1048–1066, 1993.
