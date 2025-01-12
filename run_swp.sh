
cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
decay_rate=$6

alpha_list=('7' '8' '9' '10')
cache_ratio_list=('0.1' '0.15' '0.5' '0.2')

model_path="Meta-Llama-3.1-8B-Instruct"

for alpha in "${alpha_list[@]}"; do
	for cache_ratio in "${cache_ratio_list[@]}"; do
		./run.sh ${cuda_devices} ${task} ${batch_size} ${cache_algo} ${cache_rule} ${alpha} ${cache_ratio} ${decay_rate}
	done
done

# bash run_swp.sh 3 gsm8k 1 thresholding max 0.0