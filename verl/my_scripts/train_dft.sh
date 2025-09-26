# train data generation
# python examples/data_preprocess/numina_cot.py --train_end 100000
# eval data generation
# python examples/data_preprocess/math_dataset.py
export CUDA_VISIBLE_DEVICES=1,2

nproc_per_node=2
project_name=numina-cot

lr=5e-5
# lr=7e-6


# Qwen2.5-Math-7B
# raw 5w
experiment_name=numina-cot-dft-qwen-2.5-math-7b-raw-5w-lr-$lr-0916
save_path=checkpoints/$experiment_name
model_path=../LLMs/Qwen2.5-Math-7B
train_dataset=../data/Qwen2.5-Math-7B/step1/all_data.parquet

# our data
# experiment_name=numina-cot-dft-qwen-2.5-math-7b-our-data-lr-$lr-0916
# save_path=checkpoints/$experiment_name
# model_path=../LLMs/Qwen2.5-Math-7B
# train_dataset=../data/Qwen2.5-Math-7B/model_train_datasets.parquet

# ablation
# experiment_name=numina-cot-dft-qwen-2.5-math-7b-Guided-alignmen-ablation-lr-$lr-0918
# save_path=checkpoints/$experiment_name
# model_path=../LLMs/Qwen2.5-Math-7B

# train_dataset=../data/Qwen2.5-Math-7B/ablation/Guided-alignmen/ablation_data.parquet


# Meta-Llama-3.1-8B
# experiment_name=numina-cot-dft-Meta-Llama-3.1-8B-$lr-our-data-0913
# save_path=checkpoints/$experiment_name
# model_path=../LLMs/Meta-Llama-3.1-8B

# train_dataset=../data/Meta-Llama-3.1-8B/model_train_datasets.parquet
# # train_dataset=../data/Meta-Llama-3.1-8B/step1/all_data.parquet


# train_dataset=../project/DFT/verl/data/numina_cot/train.parquet

# Meta-Llama-3.1-8B-Instruct
# raw 5w
# experiment_name=numina-cot-dft-Meta-Llama-3.1-8B-Instruct-$lr-raw-5w-0916
# save_path=checkpoints/$experiment_name
# model_path=../LLMs/Meta-Llama-3.1-8B-Instruct

# train_dataset=../data/Meta-Llama-3.1-8B-Instruct/step1/all_data.parquet

# our data
# experiment_name=numina-cot-dft-Meta-Llama-3.1-8B-Instruct-our-data-lr-$lr-0916
# save_path=checkpoints/$experiment_name
# model_path=../LLMs/Meta-Llama-3.1-8B-Instruct

# train_dataset=../data/Meta-Llama-3.1-8B-Instruct/model_train_datasets.parquet

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files=$train_dataset \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=2048 \
    optim.lr=$lr \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$model_path \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true