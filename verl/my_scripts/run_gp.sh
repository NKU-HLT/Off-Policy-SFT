
export CUDA_VISIBLE_DEVICES=7

# model_name=../LLMs/Qwen2.5-Math-7B
model_name=../LLMs/Meta-Llama-3.1-8B-Instruct

# step1

data_path=../data/Meta-Llama-3.1-8B-Instruct/step1/success_data.parquet
output_dir=../data/Meta-Llama-3.1-8B-Instruct/prob/step1

python get_probability.py \
    --model_name $model_name \
    --data_path $data_path \
    --query_column "extra_info.question" \
    --compare_column "extra_info->answer" \
    --output_path $output_dir/self.parquet \
    --device "auto"



# python get_probability.py \
#     --model_name $model_name \
#     --data_path $data_path \
#     --query_column "extra_info.question" \
#     --compare_column "extra_info.answer" \
#     --output_path $output_dir/gt.parquet \
#     --device "auto"


# step2
# data_path=../data/Meta-Llama-3.1-8B-Instruct/step2/change_cot_and_success_data.parquet
# output_dir=../data/Meta-Llama-3.1-8B-Instruct/prob/step2


# python get_probability.py \
#     --model_name $model_name \
#     --data_path $data_path \
#     --query_column "extra_info.question" \
#     --compare_column "extra_info->answer" \
#     --output_path $output_dir/change.parquet \
#     --device "auto"


# python get_probability.py \
#     --model_name $model_name \
#     --data_path $data_path \
#     --query_column "extra_info.question" \
#     --compare_column "extra_info.answer" \
#     --output_path $output_dir/gt.parquet \
#     --device "auto"


# step3
# data_path=../data/Meta-Llama-3.1-8B-Instruct/step2/gt_data.parquet
# output_dir=../data/Meta-Llama-3.1-8B-Instruct/prob/step3

# python get_probability.py \
#     --model_name $model_name \
#     --data_path $data_path \
#     --query_column "extra_info.question" \
#     --compare_column "extra_info->answer" \
#     --output_path $output_dir/gt.parquet \
#     --device "auto"