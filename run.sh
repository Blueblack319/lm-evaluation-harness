
cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
alpha=$6
cache_ratio=$7
decay_rate=$8

model_path="Meta-Llama-3.1-8B-Instruct"

CUDA_VISIBLE_DEVICES=${cuda_devices} python lm_eval/__main__.py --model hf \
	--model_args pretrained=${model_path},dtype='half' \
	--cache_args algo=${cache_algo},cache_rule=${cache_rule},alpha=${alpha},cache_ratio=${cache_ratio},decay_rate=${decay_rate} \
	--tasks ${task} --device cuda --batch_size ${batch_size} --gen_kwargs temperature=0.0 --limit 0.1
