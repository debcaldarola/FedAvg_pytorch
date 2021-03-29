#!/usr/bin/env bash

output_dir="${1:-./baseline}"

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="1000"

fedavg_lr="0.001"
# fedavg_vals: clients_per_round num_epochs
declare -a fedavg_vals=( "5 1")

###################### Functions ###################################

function move_data() {
	path="$1"
	suffix="$2"
	
	pushd models/metrics
		mv sys_metrics.csv "${path}/sys_metrics_${suffix}.csv"
		mv stat_metrics.csv "${path}/stat_metrics_${suffix}.csv"
	popd

	cp -r data/femnist/meta "${path}"
	mv "${path}/meta" "${path}/meta_${suffix}"
}

function run_fedavg() {
	clients_per_round="$1"
	num_epochs="$2"

	pushd models/
		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr} --device 'cuda' --batch-size 10
	popd
	move_data ${output_dir} "fedavg_c_${clients_per_round}_e_${num_epochs}"
}

##################### Script #################################
pushd ../

# Check that data and models are available
if [ ! -d 'data/' -o ! -d 'models/' ]; then
	echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# If data unavailable, execute pre-processing script
if [ ! -d 'data/femnist/data/train' ]; then
	if [ ! -f 'data/femnist/preprocess.sh' ]; then
		echo "Couldn't find data/ and/or models/ directories - please obtain scripts from GitHub repo: https://github.com/TalwalkarLab/leaf"
		exit 1
	fi

	echo "Couldn't find FEMNIST data - running data preprocessing script"
	pushd data/femnist/
		rm -rf meta/ data/test data/train data/rem_user_data data/intermediate
		./preprocess.sh -s niid --sf 1.00 -k 0 -t user --smplseed ${sampling_seed} --spltseed ${split_seed} --tf 0.6
	popd
fi

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
