Here it is shown how to make a basic usage of Eigen3 tools for computations related to quantum information. In particular, we implement an optimization method to achieve the optimal control field to perform a qubit-state teleportation task in a three-qubit system.

Besides that, I use this repo to track files to implement quantum optimal control using Physics-Informed Neural Networks (PINN's), keepping only those that managed to optimize all the 3 tested quantum gates using the zero-phase error.

# PINN-Based Optimal Control for Quantum Dynamics

This repository contains simulation code for applying physics-informed neural networks (PINNs) to quantum optimal control problems. The goal is to learn both the quantum state evolution and the control fields that make the system produce a desired qubit unitary operation over the span of the two lowest energy states, while preventing leakage to higher levels. Here we implemented a soft-constraints approach for the initial conditions rather than forcing them in the model.

## Overview

The method uses a composite loss function that combines:
- the residual of the Schrödinger equation,
- boundary-condition constraints at the initial and final times,
- optional penalty terms to suppress unwanted states or regularize the network.

By minimizing this loss, the neural network approximates the system dynamics while simultaneously optimizing the external control fields.

## Method

The training loop follows these steps:

1. Initialize the neural network, learning rate, loss weights, and training points.
2. Perform a forward pass to predict the state evolution and control fields.
3. Use automatic differentiation to compute time derivatives.
4. Evaluate the composite loss.
5. Backpropagate gradients and update the trainable parameters.
6. Repeat until convergence or until the maximum number of epochs is reached.

## Repository Structure

```text
.
├── src/            # Core implementation
├── notebooks/      # Example simulations and exploratory analysis
├── data/           # Input files and generated results
├── figures/        # Plots and visualizations
└── README.md
```

## Requirements

- Python 3.12+
- PyTorch 2.10.0+
- QuTiP 5.3+

## Installation

```bash
git clone https://github.com/martinjimenez388/my-first-proyect.git
cd my-first-proyect
pip install -r requirements.txt
```
Follow the instructions to install [Pytorch](https://pytorch.org/get-started/locally/#windows-installation).

## Usage

Open the example notebooks:

```bash
jupyter notebook Examples/PINN_control_GATE_3levs_FreqTrick.ipynb
jupyter notebook Examples/PINN_control_4levels_pre_trained.ipynb
```

## Loss Function

The total loss is defined as a weighted sum:

```math
L[\theta_1, \theta_2] = \sum_l c_l L_l[\theta_1, \theta_2]
```

where the trainable parameters \(\theta_1\) and \(\theta_2\) are associated with the state approximation and the control field, respectively.

The main terms are:
- `L1`: Schrödinger-equation residual.
- `L2`: boundary-condition loss.
- `L3`: penalty for forbidden subspaces.

## Example Application

This framework can be used to learn a control protocol that transfers an initial state $\(|\psi_0\rangle\)$ to a target state $\(|\psi_{\mathrm{tgt}}\rangle\)$ while avoiding undesired intermediate states.

## Results

Add figures, training curves, and example trajectories here once available.

## Citation

If you use this repository in your research, please cite the associated paper or preprint.

## License

Add your chosen license here.
