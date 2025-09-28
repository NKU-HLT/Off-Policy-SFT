
<div align="center">

# *Mind the Gap*: Data Rewriting for Stable Off-Policy <br> Supervised Fine-Tuning
  

<a href="https://arxiv.org/abs/2509.15157" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2509.15157-b31b1b.svg" height="25" />
</a>

<br>
</div>



## Abstract
Supervised fine-tuning (SFT) of large language models can be viewed as an off-policy learning problem, where expert demonstrations come from a fixed behavior policy while training aims to optimize a target policy. Importance sampling is the standard tool for correcting this distribution mismatch, but large policy gaps lead to skewed weights, high variance, and unstable optimization. Existing methods mitigate this issue with KL penalties or clipping, which passively restrict updates rather than actively reducing the gap. We propose a simple yet effective data rewriting framework that proactively shrinks the policy gap before training. For each problem, correct model-generated solutions are kept as on-policy data, while incorrect ones are rewritten through guided re-solving, falling back to expert demonstrations only when needed. This aligns the training distribution with the target policy, reducing variance and improving stability. To handle residual mismatch after rewriting, we additionally apply importance sampling during training, forming a two-stage approach that combines data-level alignment with lightweight optimization-level correction. Experiments on five mathematical reasoning benchmarks show consistent and significant gains over both vanilla SFT and the state-of-the-art Dynamic Fine-Tuning (DFT) approach.

## Code Implementation

## ⚙️ Installation

Our codebase has been tested on A800 servers with the following environment:

* `python 3.10.0`
* `torch 2.6.0+cu124`

```bash
git clone https://github.com/NKU-HLT/Off-Policy-SFT.git
cd MTG
```

### 🔧 Set Up Training Environment

```bash
conda create -n MTG python=3.10 -y
conda activate MTG
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## 🚀 Getting Started

### Step 1: Prepare Data

The dataset is stored in the link.

https://huggingface.co/datasets/xychao/MTG

### Step 2: Launch Training

```bash
./verl/my_scripts/train_dft.sh
```

### Step 3: Evaluation

To evaluate the trained model, please first follow the [Qwen2.5-Math repository](https://github.com/QwenLM/Qwen2.5-Math) to set up the evaluation environment.

```bash
# Select the prompt format matching your model
PROMPT_TYPE="qwen-boxed"
# PROMPT_TYPE="llama-base-boxed"
# PROMPT_TYPE="deepseek-math"

# Set available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configure sampling settings
N_SAMPLING=16
TEMPERATURE=1

# Specify model and output directories
MODEL_NAME_OR_PATH=""  # e.g., checkpoints/your-model-name
OUTPUT_DIR=""          # e.g., outputs/eval_results

# Run evaluation
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE
```

## Main Results

| Model / Method            | Math500   | Minerva   | Olympiad  | AIME24    | AMC23     | Avg.      |
| ------------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **Qwen2.5-Math-7B**       |           |           |           |           |           |           |
| Base                      | 39.90     | 14.43     | 17.16     | 7.50      | 29.38     | 21.67     |
| + SFT                     | 52.61     | 19.13     | 17.32     | 2.06      | 25.00     | 23.23     |
| + DFT                     | 68.70     | 31.92     | 32.31     | 6.68      | 43.44     | 36.61     |
| + DR + SFT (ours)         | 59.85     | 21.14     | 23.54     | 8.54      | 38.59     | 30.33     |
| + DR + DFT (ours)         | **70.40** | **34.85** | **36.12** | **14.58** | **54.22** | **42.03** |
|                           |           |           |           |           |           |           |
| **Llama-3.1-8B-Instruct** |           |           |           |           |           |           |
| Base                      | 36.18     | 16.01     | 9.52      | 0.83      | 14.53     | 15.41     |
| + SFT                     | 28.71     | 11.23     | 6.26      | 0.41      | 10.31     | 11.39     |
| + DFT                     | 46.50     | 24.11     | 15.65     | 3.95      | 22.50     | 22.54     |
| + DR + SFT (ours)         | 44.38     | 19.21     | 13.07     | 1.87      | 17.19     | 19.14     |
| + DR + DFT (ours)         | **47.91** | **24.72** | **16.52** | **4.99**  | **26.09** | **24.05** |

**Table**: Average accuracy (%) on mathematical reasoning benchmarks. <br> SFT = Supervised Fine-tuning, DR = Data Rewriting, <br>
DFT = Dynamic Fine-Tuning, from “On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification” [ (arXiv:2508.05629) ](https://arxiv.org/abs/2508.05629).

## Limitations
While our experiments demonstrate the effectiveness of data rewriting for stabilizing off-policy supervised fine-tuning, several limitations remain. First, our evaluation is restricted to a limited set of models, primarily at moderate parameter scales, so assessing its applicability to larger and more diverse models is left for future work. Second, we focus exclusively on mathematical reasoning benchmarks; extending the approach to broader domains, including industrial settings such as healthcare and finance, is an important next step. Third, our method adopts a single-round offline rewriting strategy, whereas more sophisticated or online approaches—e.g., rewriting per batch to mitigate policy shifts during training—could further enhance stability and performance. Finally, exploring richer rewriting techniques, such as leveraging external knowledge from more advanced models, represents another promising direction.

## Citation
If you find this paper valuable for your research or applications, we would appreciate it if you could cite our work:
```latex
@article{zhao2025mind,
  title={Mind the Gap: Data Rewriting for Stable Off-Policy Supervised Fine-Tuning},
  author={Zhao, Shiwan and Zhao, Xuyang and Zhou, Jiaming and Kong, Aobo and Li, Qicheng and Qin, Yong},
  journal={arXiv preprint arXiv:2509.15157},
  year={2025}
}
```


