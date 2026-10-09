# Perceptron Reconstructed

A reproducible, interactive research project exploring both the classic 1958 Rosenblatt Perceptron model and a modern Scikit-Learn based machine learning pipeline.

## Overview

The Perceptron Reconstructed project provides a unified environment for running and analyzing linear classification algorithms. It includes a frontend interactive dashboard that connects to a fast Python backend. The system allows users to experiment with theoretical concepts, visualize the neural network architecture dynamically, and execute modern data science workflows.

## Core Features

* **Interactive Neural Network Interface**: Real time SVG visualization of the input, hidden, and output layers as parameters are modified.
* **1958 Model Simulation**: A classic probabilistic implementation demonstrating state space expansion and the bivalent gamma reinforcement rule.

## Project Structure

* `api.py`: The FastAPI backend serving the simulation execution.
* `frontend/`: The React based user interface for configuring and visualizing experiments.
* `backend/`: The core Python logic for the classic Rosenblatt simulation.

## Usage Instructions

1. Start the backend server:
```bash
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. Start the frontend application:
```bash
cd frontend
bun run dev
```

Navigate to the provided localhost URL to interact with the system.

## License

Copyright (c) 2026. Licensed under the Apache License 2.0.
