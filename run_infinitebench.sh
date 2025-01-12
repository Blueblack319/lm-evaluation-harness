cuda_devices=$1
task=$2
batch_size=$3
cache_algo=$4
cache_rule=$5
alpha=$6
cache_ratio=$7
decay_rate=$8
max_seq_length=$9
num_eval_examples=${10}
attn_type=${11}

model_path="Llama-3-8B-Instruct-262k"

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")

export TOKENIZERS_PARALLELISM=false

for task in ${TASKS[@]}; do
echo $task
CUDA_VISIBLE_DEVICES=${cuda_devices} python lm_eval/__main__.py --model hf \
	--model_args pretrained=${model_path},dtype='half' \
	--cache_args algo=${cache_algo},cache_rule=${cache_rule},alpha=${alpha},cache_ratio=${cache_ratio},decay_rate=${decay_rate} \
	--tasks ${task} --device cuda --batch_size ${batch_size} --gen_kwargs temperature=0.0 --limit 0.1 --is_infinitebench \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length ${max_seq_length} \
    --rewrite \
    --num_eval_examples ${num_eval_examples} --topk 1 --starting_layer 0 --attn_type ${attn_type}
done

# ./run_infinitebench.sh 3 gsm8k 1 ideal_2_upper max 7 1.0 0.0 160000 -1 hf