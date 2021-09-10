#!/usr/bin/env bash

output_dir="${1:-./baseline}"

num_rounds="50000"

fedavg_lr="0.001"
# fedavg_vals: clients_per_round num_epochs
batch_size ="32"
declare -a fedavg_vals=( "10 1")

###################### Functions ###################################

function run_fedavg() {
	clients_per_round="$1"
	num_epochs="$2"

	pushd models/
		python main.py -dataset 'inaturalist' -model 'mobilenet' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr} --batch-size 32 --eval-every 2500 -device 'cuda:2' --weight-decay 0.00004
	popd
}

##################### Script #################################
pushd ../

# Create output_dir
mkdir -p ${output_dir}
output_dir=`realpath ${output_dir}`
echo "Storing results in directory ${output_dir} (please invoke this script as: ${0} <dirname> to change)"


# Run FedAvg experiments
for val_pair in "${fedavg_vals[@]}"; do
	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
	num_epochs=`echo ${val_pair} | cut -d' ' -f2`
	echo "Running FedAvg experiment with ${num_epochs} local epochs and ${clients_per_round} clients"
	run_fedavg "${clients_per_round}" "${num_epochs}"
done

popd
