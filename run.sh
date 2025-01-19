
cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
alpha=$6
cache_ratio=$7
decay_rate=$8
eff_threshold=$9
limit=${10}

model_path="Meta-Llama-3.1-8B-Instruct"

CUDA_VISIBLE_DEVICES=${cuda_devices} python lm_eval/__main__.py --model hf \
	--model_args pretrained=${model_path},dtype='half' \
	--cache_args algo=${cache_algo},cache_rule=${cache_rule},alpha=${alpha},cache_ratio=${cache_ratio},decay_rate=${decay_rate},eff_threshold=${eff_threshold} \
	--tasks ${task} --device cuda --batch_size ${batch_size} --gen_kwargs temperature=0.0 --limit ${limit} --verbosity DEBUG

# bash run.sh 3 gsm8k 1 ideal_2_upper max 7 0.1 0.0 0.9 0.1