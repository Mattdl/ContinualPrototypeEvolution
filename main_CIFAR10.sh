#!/bin/bash
MY_PYTHON=python
pyscript=main.py
exp_name="demo_CIFAR10"

# Paths
datapath="./data"
ofile="cifar10.pt"

# Main Split-CIFAR10 config
results="results/cifar10/"
ds_args="--tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 100 --samples_per_task 10000 --data_file $ofile --cuda yes"

# Appendix Split-CIFAR10-mini config
#results="results/cifar10mini/"
#ds_args="--tasks_to_preserve 5 --save_path $results --data_path $datapath --log_every 100 --samples_per_task 2000 --data_file $ofile --cuda yes"
mkdir 'results'
mkdir $results

cd "$datapath" || exit
cd raw/ || exit
$MY_PYTHON raw.py 'cifar10' # Download
cd ..

# Prepare dataset
if [ ! -f $ofile ]; then
  echo "Preprocessing $ofile"
  $MY_PYTHON "cifar10.py" \
    --o $ofile \
    --i "raw/cifar10.pt" \
    --seed 0 \
    --n_tasks 5
fi
cd ..

##########################################################
# BALANCED method configs
##########################################################
# Methods: 'prototypical.CoPE', 'CoPE_CE', 'finetune', 'reservoir',  'gem', 'icarl', 'GSSgreedy'

# Grid over (pick best)
# n_outputs=128,256
# lr=0.05,0.01,0.005,0.001
n_memories=100 # Change for ablation (mem per class): 10,20,50,100,150,200

# CoPE
# Last code version acc check for 5 seeds: 49.610+-3.441 avg acc
# Avg accs per seed=[tensor([0.4576]), tensor([0.5106]), tensor([0.5040]), tensor([0.4664]), tensor([0.5419])]
# Note: Use the exact dependencies in README.md to reproduce the results.
model="prototypical.CoPE"
args="--model $model --batch_size 10 --lr 0.005 --loss_T 0.1 --p_momentum 0.99 --n_memories $n_memories --n_outputs 256 --n_iter 1 --n_seeds 5 $exp_name"
$MY_PYTHON "$pyscript" $ds_args $args # Run python file

# Cope-CE
model="CoPE_CE"
args="--model $model --batch_size 10 --lr 0.05 --n_memories $n_memories --n_outputs 10 --n_iter 1 --n_seeds 5 $exp_name"

# iid-offline
model="finetune"
args="--n_iter 1 --n_epochs 50 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# iid-online
model="finetune"
args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid yes $exp_name"

# finetune
model="finetune"
args="--n_iter 1 --n_epochs 1 --model $model --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 --iid no $exp_name"

# Reservoir
model="reservoir"
n_mem_tot=1000
args="--n_iter 1 --model $model --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# iCaRL
n_mem_tot=1000
model="icarl"
args="--n_iter 1 --model $model --memory_strength 1 --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 ICARL"

# GEM
model="gem"
n_mem_task=200 # 5 tasks
args="--n_iter 1 --model $model --memory_strength 0.5 --n_memories $n_mem_task --lr 0.005 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# GSS
model="GSSgreedy"
n_mem_tot=1000
args="--model $model --batch_size 10 --lr 0.05 --n_memories 10 --n_sampled_memories $n_mem_tot --n_constraints 10 --memory_strength 10 --n_iter 1 --change_th 0. --subselect 1 --normalize no $exp_name"

# MIR
# See original implementation @ https://github.com/optimass/Maximally_Interfered_Retrieval
# Adapted to match settings in this paper

exit

##########################################################
# IMBALANCED method configs
##########################################################
# Gridsearch over (pick best)
# Same hyperparams as low-capacity setups in Appendix.
# n_outputs=128,256
# lr=0.05,0.01,0.005,0.001 -> Pick best for each S(T_i)
# n_iter=1,5
n_memories=100 # 1k total

# CoPE
model="prototypical.CoPE"
args="--model $model --batch_size 10 --lr 0.05 --loss_T 0.1 --p_momentum 0.99 --n_memories $n_memories --n_outputs 128 --n_iter 5 --n_seeds 5 $exp_name"

# S(T_i), for i in (1 to 5) -> Tasks 1 to 5
extra_args="--samples_per_task |1,4000,400|" # First task has 4000, rest has 400
extra_args="--samples_per_task |2,4000,400|"
extra_args="--samples_per_task |3,4000,400|"
extra_args="--samples_per_task |4,4000,400|"
extra_args="--samples_per_task |5,4000,400|"      # Fifth task has 4000, rest has 400

$MY_PYTHON "$pyscript" $ds_args $extra_args $args # Run python file
