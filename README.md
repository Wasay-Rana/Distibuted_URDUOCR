# Distributed URDU OCR Training Project

## Overview
This project aims to train a URDU OCR (Optical Character Recognition) model using distributed computing across multiple nodes, each equipped with NVIDIA RTX 4070 GPUs. The project is designed to handle a large dataset of 35 million images of Urdu words up to length 5.

**Note: This project is currently in development and requires further testing and refinement.**

## System Requirements
- 3 nodes, each with:
  - NVIDIA RTX 4070 GPU
  - 128GB DDR4 RAM
  - 2TB SSD
- Shared file system through NFS
- PyTorch and related dependencies (specific versions TBD)
- CUDA-compatible environment

## Project Structure
- `model.py`: Defines the URDU OCR neural network model
- `dataset.py`: Handles data loading and preprocessing
- `train.py`: Main script for distributed training
- `run_distributed.sh`: Shell script to launch distributed training across nodes

## Setup and Installation
1. Ensure all nodes have the required hardware and software setup.
2. Install PyTorch and other necessary Python libraries (requirements.txt to be added).
3. Set up the shared NFS file system across all nodes.
4. Clone this repository to all nodes.

## Usage
1. Modify the `DATA_DIR` in `run_distributed.sh` to point to your NFS-shared dataset.
2. Update the `MASTER_ADDR` in `run_distributed.sh` with the IP address of your master node.
3. Adjust the `num_classes` in `train.py` to match your specific URDU OCR task.
4. Make the shell script executable:
	chmod +x run_distributed.sh
5. Run the script on each node:
	./run_distributed.sh
## Current Limitations and TODOs
- The model architecture in `model.py` is a placeholder and needs to be optimized for URDU OCR.
- Data loading in `dataset.py` requires implementation of proper labeling logic.
- The training script (`train.py`) needs additional features such as checkpointing and validation.
- Hyperparameters need to be tuned for optimal performance.
- Error handling and logging need to be improved.
- The setup has not been thoroughly tested on the specified hardware configuration.

## Contributing
Contributions to improve and complete this project are welcome. Please submit pull requests or open issues to discuss proposed changes or report bugs.

## Disclaimer
This project is a work in progress and may contain errors or inefficiencies. Use at your own risk and always back up your data before running large-scale training operations.
