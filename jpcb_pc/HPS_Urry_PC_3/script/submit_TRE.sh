#!/bin/bash
#SBATCH --job-name=urry
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --array=0-24
#SBATCH --nodes=10
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --mem=3000M
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/TRE_%a.txt

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "SLURM_CPUS_ON_NODE" $SLURM_CPUS_ON_NODE

nodelist=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "node list: $nodelist"

nodes_array=($nodelist)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" sleep 1 > /dev/null 2>&1; hostname --ip-address)
echo "head node: $head_node"
echo "head node ip: $head_node_ip"

port=$(expr 6030 + $SLURM_ARRAY_TASK_ID)
head_node_ip_with_port=$head_node_ip:$port

sleep $(( 1 + $RANDOM % 60 ))

echo "starting head node"
srun --nodes=1 --ntasks=1 --nodelist="$head_node" \
     ray start --head --block \
     --num-cpus=$SLURM_CPUS_ON_NODE \
     --num-gpus=1 \
     --port=$port \
     --include-dashboard=False &
sleep $(( 1 + $RANDOM % 60 ))

echo "starting worker nodes"
for i in $(seq 1 $(( $SLURM_JOB_NUM_NODES-1 )) ); do
    node=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node"
    srun --nodes=1 --ntasks=1 --nodelist=$node \
         ray start --address=$head_node_ip_with_port --block \
         --num-cpus=$SLURM_CPUS_ON_NODE \
         --num-gpus=1 &

done

export RAY_ADDRESS=$head_node_ip_with_port

PYTHON=/home/xqding/apps/miniconda3/envs/jop/bin/python
#$PYTHON -u ./script/run_TRE.py --idx_protein $SLURM_ARRAY_TASK_ID --type single
$PYTHON -u ./script/run_TRE.py --idx_protein $SLURM_ARRAY_TASK_ID --type double


     # --ray-client-server-port=6293 \
     # --node-manager-port=6497 \
     # --object-manager-port=6604 \
     # --runtime-env-agent-port=6862 \
     # --min-worker-port=10000 \
     # --max-worker-port=12000 \

     #          --node-manager-port=6497 \
     #     --object-manager-port=6604 \
     #     --runtime-env-agent-port=6862 &
