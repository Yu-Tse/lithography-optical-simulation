# Lithography Optical Simulation

This project provides a simulation framework for analyzing photolithographic mask patterns and their interaction with customized light sources using Fourier optics. It includes visualizations and statistical evaluations, and is ideal for academic research or semiconductor process exploration.

## Features

- Generate complex photomask layouts
- Create customizable quasar-like illumination sources
- Simulate aerial image formation using FFT
- Visualize 2D and 3D intensity distributions
- Calculate statistical metrics (standard deviation)
- Prepare dataset for optimization and AI applications

## File Structure

- `litho.py` - Single-shot mask and light source simulation with visualization.
- `lithoData.py` - Batch simulation for multiple light source parameters, saving statistical results to CSV.
- `lithoOpt.py` - Optimizes light source parameters to minimize intensity irregularities using `basinhopping`.

## Installation

```bash
git clone https://github.com/Yu-Tse/lithography-optical-simulation.git
cd lithography-optical-simulation
pip install -r requirements.txt
