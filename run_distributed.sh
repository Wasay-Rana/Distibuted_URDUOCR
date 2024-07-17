#!/bin/bash

# Set the number of nodes
NUM_NODES=3

# Set the number of GPUs per node
GPUS_PER_NODE=1

# Calculate the world size
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Set the master node address (replace with actual IP)
MASTER_ADDR="192.168.1.1"

# Set the master port
MASTER_PORT=29500

# Set the data directory (replace with actual path)
DATA_DIR="/path/to/your/data"

# Loop through all nodes
for NODE_RANK in $(seq 0 $((NUM_NODES-1)))
do
    # If this is the current node (you'll need to run this script on each node)
    if [ "$NODE_RANK" -eq "$SLURM_NODEID" ]; then
        python -m torch.distributed.launch \
            --nproc_per_node=$GPUS_PER_NODE \
            --nnodes=$NUM_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            train.py \
            --batch-size 64 \
            --num-epochs 10 \
            --learning-rate 0.001 \
            --num-classes 1000 \
            --data-dir $DATA_DIR
    fi
done
