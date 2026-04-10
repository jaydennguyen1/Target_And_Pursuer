# Target_And_Pursuer
3D Python simulation of missile pursuit-evasion dynamics, featuring randomized evasive maneuvers, advanced decoy countermeasures, and real-time telemetry visualization for analyzing engagement outcomes.

# Pursuit-Evasion Guidance Simulation

This repository contains a 3D simulation developed in Python to model complex pursuit-evasion scenarios, specifically focusing on missile guidance against an evasive target employing countermeasures. The simulation provides a robust framework for analyzing the effectiveness of different evasion strategies and countermeasure deployments in a dynamic environment.

## Features

*   **3D Kinematic Simulation:** Models the movement of a pursuer (missile) and a target (aircraft) in a three-dimensional space.
*   **Randomized Evasive Maneuvers:** The target aircraft executes randomized evasive turns with configurable durations and angles, simulating unpredictable flight patterns.
*   **Advanced Decoy Countermeasures:** Implements a sophisticated decoy system where the target releases distractions with specific flight characteristics, designed to divert the pursuer.
*   **Dynamic Target Selection:** The pursuer's guidance system dynamically chooses between tracking the primary target or a decoy based on proximity and age of the distraction, incorporating a configurable bias.
*   **Configurable Parameters:** Easily adjust simulation parameters such as velocities, launch times, kill distances, turn characteristics, and decoy properties to explore various scenarios.
*   **Real-time Telemetry Visualization:** Utilizes `matplotlib` to provide a live 3D plot of the pursuer, target, and active decoys, offering immediate visual feedback on engagement dynamics.
*   **Data Logging:** Records key simulation data, including positions, velocities, and tracking decisions, for post-analysis.

## Technical Details

The simulation is built upon a discrete-time integration scheme (`dt = 0.01s`) over a specified maximum time (`tmax = 300s`).

*   **Pursuer Guidance:** The missile employs a proportional navigation-like guidance law, continuously adjusting its trajectory to intercept the currently tracked object.
*   **Target Evasion Logic:** The target's evasion is governed by a pre-defined, randomized turn plan. When the pursuer closes within a `turn_trigger_distance`, the target initiates a sequence of evasive maneuvers, each comprising a turn phase and a straight flight phase.
*   **Decoy Mechanics:** Decoys are spawned at the target's position, moving at a fraction of the target's speed and with a lateral spread. Their effectiveness is modeled by a scoring function that considers distance from the pursuer and age, influencing the pursuer's tracking decision.
*   **Motion Smoothing:** Evasive turns incorporate a smooth transition function (6th-order polynomial) to ensure realistic acceleration profiles.

## Installation

To run this simulation, you need Python 3.x and the following libraries:

```bash
pip install numpy matplotlib
```

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/pursuit-evasion-simulation.git
    cd pursuit-evasion-simulation
    ```
2.  Run the simulation script:
    ```bash
    python your_simulation_script_name.py
    ```
    (Note: Replace `your_simulation_script_name.py` with the actual filename of your Python code.)

3.  The simulation will output a 3D animated plot showing the trajectories of the missile, target, and decoys. Console output will provide details on key events such as missile launch, evasive maneuvers, and interception.

## Configuration

Key simulation parameters can be adjusted directly within the `VARIABLES` section of the Python script:

*   `targ_vel`, `miss_vel`: Velocities of the target and missile.
*   `turn_trigger_distance`: Distance at which the target begins evasive maneuvers.
*   `min_num_turns`, `max_num_turns`: Range for the number of evasive turns.
*   `distraction_enabled`: Boolean to enable/disable decoy deployment.
*   `distraction_release_interval`, `distraction_lifetime`: Parameters for decoy release and duration.
*   `distraction_switch_bias`, `distraction_age_penalty`: Factors influencing the pursuer's decision to track decoys.

## Future Enhancements

*   Implement more sophisticated guidance laws (e.g., Proportional Navigation with Lead).
*   Introduce environmental factors (wind, atmospheric density).
*   Develop a graphical user interface (GUI) for real-time parameter adjustment.
*   Expand analysis capabilities with statistical summaries of multiple simulation runs.
*   Integrate different types of countermeasures and evasion tactics.

