
cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
alpha=$6
cache_ratio=$7
decay_rate=$8
eff_threshold=$9
prefill_algo=${10}
prefill_ratio=${11}
block_num=${12}
anchor_num=${13}
limit=${14}

model_path="Meta-Llama-3.1-8B-Instruct"

CUDA_VISIBLE_DEVICES=${cuda_devices} python lm_eval/__main__.py --model hf \
	--model_args pretrained=${model_path},dtype='half' \
	--cache_args algo=${cache_algo},cache_rule=${cache_rule},alpha=${alpha},cache_ratio=${cache_ratio},decay_rate=${decay_rate},eff_threshold=${eff_threshold},prefill_algo=${prefill_algo},prefill_ratio=${prefill_ratio},block_num=${block_num},anchor_num=${anchor_num} \
	--tasks ${task} --device cuda --batch_size ${batch_size} --gen_kwargs temperature=0.0 --limit ${limit} --verbosity DEBUG

# bash run_starlink.sh {cuda_devices} {task} {batch_size} {cache_algo} {cache_rule} {alpha} {cache_ratio} {decay_rate} {eff_threshold} {prefill_algo} {prefill_ratio} {block_num} {anchor_num} {limit}
# bash run_starlink.sh 3 gsm8k 1 ideal_2_upper max 7 1.0 0.0 0.9 full 5 5 0.1
# bash run_starlink.sh 3 gsm8k 1 ideal_2_upper max 7 1.0 0.0 0.9 starlink_upper_1 0.5 5 5 0.1