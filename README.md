# PSC_Robot

PSC_Robot is a two-wheeled balancing robot simulation project built with MuJoCo and a PySide6 desktop interface. It supports LQR, neural network, and reinforcement learning controllers, with tools for training and evaluation.

## Features

- **LQR:** The default controller for balancing and movement.
- **Neural network:** A supervised controller trained to imitate LQR, with FP32 and INT8 inference options.
- **Reinforcement learning:** PPO training for balance and velocity tracking.
- **Interactive controls:** Adjust speed and yaw manually, reset the robot, or use AUTO mode to run predefined command patterns.

## Repository layout

- `balance-robot-mujoco-sim/`: Simulator, training tools, and evaluation scripts.
- `balance-robot-mujoco-sim/models/`: Saved models and evaluation results.
- `balance-robot-mujoco-sim/tests/`: Automated tests and GUI checks.
- `external/PSC-ONE/`: PSC-ONE hardware and software submodule.

## Getting started

Install Conda and use a desktop environment with OpenGL support. From the repository root, create the environment and launch the simulator:

```bash
cd balance-robot-mujoco-sim
conda env create -f conda-environment.yaml
conda activate robot-mujoco-simulation
python src/simulation/simulate_robot.py
```

The simulator starts in LQR / MANUAL mode. Use the Speed and Yaw sliders to control movement. Reset restores the robot while retaining the selected controller.

If you need the PSC-ONE sources, initialize the submodule from the repository root:

```bash
git submodule update --init --recursive
```

## Optional controllers

Install the dependencies for the features you need from `balance-robot-mujoco-sim/`, with the Conda environment active:

```bash
python -m pip install -r requirements-nn.txt    # Supervised NN training
python -m pip install -r requirements-rl.txt    # PPO training and control
python -m pip install -r requirements-int8.txt  # INT8 inference
```

LQR and exported FP32 neural network inference only require the base environment. Use the GUI training panels to train models and view progress. For a saved RL model, use **Load RL** before selecting **Reinforcement Learning**. AUTO mode runs command patterns for evaluation; it does not update model weights.

## Documentation

See the simulator documentation in [English](balance-robot-mujoco-sim/README.md) or [Japanese](balance-robot-mujoco-sim/README_JP.md) for training commands, model formats, and evaluation details. INT8 results are documented in the [validation report](balance-robot-mujoco-sim/docs/int8-validation.md).

## License

[MIT](LICENSE). See the included projects for their respective license terms.
