#!/bin/bash

# Run the experiment
# Accuracy experiments on integer quantized models 
METHODS=('cosplaces2048int8' 'eigenplaces2048int8' 'dinov2_boqint8' 'dinov2_saladint8' 'resnet50_boqint8')
DATASETS=('msls' 'essex3in1' 'pitts30k' 'tokyo247' 'svox-sun' 'svox-rain' 'svox-night' 'svox-snow')

# Loop over all combinations of methods and datasets
for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running experiment with method: $method, dataset: $dataset"
        python run_experiment.py --method "$method" --dataset "$dataset" --metrics r1 r5 r10
    done
done


# DB Memory experiments on integer quantized models 
METHODS=('cosplaces2048int8' 'eigenplaces2048int8' 'dinov2_boqint8' 'dinov2_saladint8' 'resnet50_boqint8')
DATASETS=('msls' 'essex3in1' 'pitts30k' 'tokyo247' 'svox-sun' 'svox-rain' 'svox-night' 'svox-snow')

# Loop over all combinations of methods and datasets
for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running experiment with method: $method, dataset: $dataset"
        python run_experiment.py --method "$method" --dataset "$dataset" --metrics model_mem db_mem
    done
done


# ==================== ALL Latency Experiments and memory ====================

METHODS=(
    'cosplaces32'
    'cosplaces64'
    'cosplaces128'
    'cosplaces512'
    'cosplaces1024'
    'cosplaces2048'
    'cosplaces2048int8'
    'eigenplaces128'
    'eigenplaces256'
    'eigenplaces512'
    'eigenplaces2048'
    'eigenplaces2048int8'
    'dinov2_boq'
    'dinov2_salad'
    'dinov2_boqint8'
    'dinov2_saladint8'
    'resnet50_boq'
    'resnet50_boqint8'
    'netvlad_sp'
    'tetra-boq-dd1'
    'tetra-boq-dd2'
    'tetra-gem-dd1'
    'tetra-gem-dd2'
    'tetra-salad-dd1'
    'tetra-salad-dd2'
    'tetra-mixvpr-dd1'
    'tetra-mixvpr-dd2'
)

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running experiment with method: $method, dataset: $dataset"
        python run_experiment.py --method "$method" --dataset "$dataset" --metrics gpu_time match_time
    done
done


for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running experiment with method: $method, dataset: $dataset"
        python run_experiment.py --method "$method" --dataset "$dataset" --metrics model_mem db_mem
    done
done