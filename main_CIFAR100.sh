#!/bin/bash
MY_PYTHON=python
pyscript=main.py
exp_name="demo_CIFAR100"

# Paths
datapath="./data"
ofile="cifar100.pt"

# Main Split-CIFAR100 config
tasks=20
results="results/cifar100/"
ds_args="--tasks_to_preserve $tasks --save_path $results --data_path $datapath --log_every 500 --samples_per_task 2500 --data_file $ofile --cuda yes"
mkdir 'results'
mkdir $results

cd "$datapath" || exit
cd raw/ || exit
$MY_PYTHON raw.py 'cifar100' # Download
cd ..

# Prepare dataset
if [ ! -f $ofile ]; then
  echo "Preprocessing $ofile"
  $MY_PYTHON "cifar100.py" \
    --o $ofile \
    --i "raw/cifar100.pt" \
    --seed 0 \
    --n_tasks $tasks
fi
cd ..

##########################################################
# BALANCED method configs
##########################################################
# Methods: 'prototypical.CoPE', 'CoPE_CE', 'finetune', 'reservoir',  'gem', 'icarl', 'GSSgreedy'

# Gridsearch over (pick best)
# n_outputs=128,256,512
# lr=0.05,0.01,0.005,0.001
# n_iter=1,5
n_memories=500 # Change for ablation (mem per class)

# CoPE
model="prototypical.CoPE"
args="--model $model --batch_size 10 --lr 0.005 --loss_T 0.05 --p_momentum 0.9 --n_memories $n_memories --n_outputs 256 --n_iter 5 --n_seeds 5 $exp_name"
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
n_mem_tot=5000
args="--n_iter 1 --model $model --n_memories $n_mem_tot --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# iCaRL
n_mem_tot=5000
model="icarl"
args="--n_iter 5 --model $model --memory_strength 1 --n_memories $n_mem_tot --lr 0.01 --batch_size 10 --n_outputs 10 --n_seeds 5 ICARL"

# GEM
model="gem"
n_mem_task=250 # 20 tasks
args="--n_iter 5 --model $model --memory_strength 0.5 --n_memories $n_mem_task --lr 0.05 --batch_size 10 --n_outputs 10 --n_seeds 5 $exp_name"

# GSS
model="GSSgreedy"
n_mem_tot=5000
args="--model $model --batch_size 10 --lr 0.05 --n_memories 10 --n_sampled_memories $n_mem_tot --n_constraints 10 --memory_strength 10 --n_iter 5 --change_th 0. --subselect 1 --normalize no $exp_name"

# MIR
# See original implementation @ https://github.com/optimass/Maximally_Interfered_Retrieval
# Adapted to match settings in this paper

exit

##########################################################
# IMBALANCED method configs
##########################################################
# Gridsearch over (pick best)
# Same hyperparams as low-capacity setups in Appendix.
# n_outputs=256
# lr=0.05,0.01,0.005,0.001
# n_iter=1,5
n_memories=500 # 5k total

# CoPE
model="prototypical.CoPE"
args="--model $model --batch_size 10 --lr 0.05 --loss_T 0.05 --p_momentum 0.9 --n_memories $n_memories --n_outputs 256 --n_iter 5 --n_seeds 5 $exp_name"

# S(T_i), for i in (1 to 5) -> Tasks 1,5,10,15,20
extra_args="--samples_per_task |1,2500,1000|"     # First task has 2500 samples, all the rest 1000
extra_args="--samples_per_task |5,2500,1000|"     # Fifth task has 2500 samples, all the rest 1000
extra_args="--samples_per_task |10,2500,1000|"    # Tenth task has 2500 samples, all the rest 1000
extra_args="--samples_per_task |15,2500,1000|"    # 15th task has 2500 samples, all the rest 1000
extra_args="--samples_per_task |20,2500,1000|"    # 20th task has 2500 samples, all the rest 1000

$MY_PYTHON "$pyscript" $ds_args $extra_args $args # Run python file
