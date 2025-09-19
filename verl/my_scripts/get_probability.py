import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundtruthProbabilityCalculator:
    def __init__(self, model_name: str, device: str = "auto"):
        """
        初始化概率计算器
        
        Args:
            model_name: Hugging Face模型名称
            device: 设备选择 ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # 加载模型和tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # 如果没有pad_token，设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def calculate_sequence_probability(self, query: str, groundtruth: str) -> Dict:
        """
        计算给定query下生成groundtruth的概率
        
        Args:
            query: 输入查询
            groundtruth: 真实答案
            
        Returns:
            包含各种概率指标的字典
        """
        # 构建完整的输入序列
        full_text = query + groundtruth
        
        # Tokenize
        query_tokens = self.tokenizer.encode(query, return_tensors="pt", add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=True)
        
        query_length = query_tokens.shape[1]
        total_length = full_tokens.shape[1]
        groundtruth_length = total_length - query_length
        
        if groundtruth_length <= 0:
            logger.warning("Groundtruth is empty or too short")
            return self._empty_result()
        
        # 移动到设备
        full_tokens = full_tokens.to(self.device)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(full_tokens)
            logits = outputs.logits
            
            # 计算groundtruth部分的概率
            # logits的形状是 [batch_size, sequence_length, vocab_size]
            # 我们需要logits[:-1]来预测tokens[1:]
            target_logits = logits[0, query_length-1:total_length-1]  # groundtruth部分的logits
            target_tokens = full_tokens[0, query_length:total_length]   # groundtruth部分的tokens
            
            # 计算每个token的概率
            probs = F.softmax(target_logits, dim=-1)
            log_probs = F.log_softmax(target_logits, dim=-1)
            
            # 获取目标token的概率
            target_probs = probs[range(groundtruth_length), target_tokens]
            target_log_probs = log_probs[range(groundtruth_length), target_tokens]
            
            # 计算各种概率指标
            joint_log_prob = target_log_probs.sum().item()
            joint_prob = torch.exp(torch.tensor(joint_log_prob)).item()
            
            # 几何平均概率（平均每个token的概率）
            geometric_mean_prob = torch.exp(target_log_probs.mean()).item()
            
            # 困惑度
            perplexity = torch.exp(-target_log_probs.mean()).item()
            
            # Token级别的概率
            token_probs = target_probs.cpu().numpy()
            token_log_probs = target_log_probs.cpu().numpy()
            
            # 解码tokens用于调试
            groundtruth_tokens = [self.tokenizer.decode([token]) for token in target_tokens]
        
        return {
            'joint_probability': joint_prob,
            'joint_log_probability': joint_log_prob,
            'geometric_mean_probability': geometric_mean_prob,
            'perplexity': perplexity,
            'groundtruth_length': groundtruth_length,
            'token_probabilities': token_probs.tolist(),
            'token_log_probabilities': token_log_probs.tolist(),
            'tokens': groundtruth_tokens,
            'query': query,
            'groundtruth': groundtruth
        }
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'joint_probability': 0.0,
            'joint_log_probability': float('-inf'),
            'geometric_mean_probability': 0.0,
            'perplexity': float('inf'),
            'groundtruth_length': 0,
            'token_probabilities': [],
            'token_log_probabilities': [],
            'tokens': [],
            'query': '',
            'groundtruth': ''
        }
    
    def process_dataset(self, 
                       parquet_path: str, 
                       query_column: str, 
                       compare_column: str,
                       batch_size: int = 1,
                       max_samples: Optional[int] = None) -> List[Dict]:
        """
        处理parquet数据集
        
        Args:
            parquet_path: parquet文件路径
            query_column: 查询列名
            compare_column: 对比的列名
            batch_size: 批处理大小（目前只支持1）
            max_samples: 最大处理样本数
            
        Returns:
            结果列表
        """
        logger.info(f"Loading dataset from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # if query_column not in df.columns or groundtruth_column not in df.columns:
        #     raise ValueError(f"Columns {query_column} or {groundtruth_column} not found in dataset")
        
        if max_samples:
            df = df.head(max_samples)
        
        results = []
        
        logger.info(f"Processing {len(df)} samples")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating probabilities"):
            query = str(row[query_column])
            if "->" in compare_column:
                # 支持多列对比
                compare_cols = [col.strip() for col in compare_column.split("->")]
                compare_str = str(row[compare_cols[0]][compare_cols[1]])
            else:
                compare_str = str(row[compare_column])
            
            try:
                result = self.calculate_sequence_probability(query, compare_str)
                result['sample_id'] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                error_result = self._empty_result()
                error_result['sample_id'] = idx
                error_result['error'] = str(e)
                results.append(error_result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存结果到文件"""
        results_df = pd.DataFrame(results)
        
        if output_path.endswith('.parquet'):
            results_df.to_parquet(output_path, index=False)
        elif output_path.endswith('.csv'):
            results_df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            results_df.to_json(output_path, orient='records', indent=2)
        else:
            # 默认保存为parquet
            results_df.to_parquet(output_path + '.parquet', index=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate LLM groundtruth probabilities")
    parser.add_argument("--model_name", required=True, help="Hugging Face model name")
    parser.add_argument("--data_path", required=True, help="Path to parquet dataset")
    parser.add_argument("--query_column", required=True, help="Column name for queries")
    parser.add_argument("--compare_column", required=True, help="Column name for compare")
    parser.add_argument("--output_path", required=True, help="Output file path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only supports 1)")
    
    args = parser.parse_args()
    
    # 初始化计算器
    calculator = GroundtruthProbabilityCalculator(args.model_name, args.device)
    
    # 处理数据集
    results = calculator.process_dataset(
        args.data_path,
        args.query_column,
        args.compare_column,
        args.batch_size,
        args.max_samples
    )
    
    # 保存结果
    calculator.save_results(results, args.output_path)
    # # 打印统计信息
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        avg_joint_prob = np.mean([r['joint_probability'] for r in valid_results])
        avg_joint_log_prob = np.mean([r['joint_log_probability'] for r in valid_results])
        avg_geometric_mean_prob = np.mean([r['geometric_mean_probability'] for r in valid_results])
        avg_perplexity = np.mean([r['perplexity'] for r in valid_results])
        logger.info(f"Average joint probability: {avg_joint_prob:.6f}")
        logger.info(f"Average joint log probability: {avg_joint_log_prob:.2f}")
        logger.info(f"Average geometric mean probability: {avg_geometric_mean_prob:.6f}")
        logger.info(f"Average perplexity: {avg_perplexity:.2f}")
        logger.info(f"Successfully processed: {len(valid_results)}/{len(results)} samples")

if __name__ == "__main__":
    main()