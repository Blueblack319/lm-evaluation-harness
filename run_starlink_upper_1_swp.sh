
cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
alpha=$6
# cache_ratio=$7
decay_rate=$7
eff_threshold=$8
prefill_algo=$9
block_num=${10}
anchor_num=${11}
limit=${12}

prefill_ratio_list=('0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0')
# cache_ratio_list=('0.9' '0.7' '0.5' '0.3' '0.1')
cache_ratio_list=('0.8' '0.6' '0.4' '0.2' '0.05')

model_path="Meta-Llama-3.1-8B-Instruct"
for cache_ratio in "${cache_ratio_list[@]}"; do
	for prefill_ratio in "${prefill_ratio_list[@]}"; do
		bash run_starlink.sh ${cuda_devices} ${task} ${batch_size} ${cache_algo} ${cache_rule} ${alpha} ${cache_ratio} ${decay_rate} ${eff_threshold} ${prefill_algo} ${prefill_ratio} ${block_num} ${anchor_num} ${limit}
	done
done
# bash run_starlink_upper_1_swp.sh 3 gsm8k 1 ideal_2_upper max 7 1.0 0.0 0.9 starlink_upper_1 5 5 0.1
# bash run_starlink_upper_1_swp.sh 3 gsm8k 1 ideal_2_upper max 7 0.0 0.9 starlink_upper_1 5 5 0.1