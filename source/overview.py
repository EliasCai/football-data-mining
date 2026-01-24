import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class MatchOverview:
    """
    足球赔率数据处理器：合并原始数据、计算比赛结果、转换赔率概率、输出分析结果
    {
    "schema": {
        "fields": [
        {
            "name": "期数id",
            "type": "integer",
            "description": "彩票期的唯一标识符。"
        },
        {
            "name": "比赛id",
            "type": "string",
            "description": "每场比赛的唯一标识符。"
        },
        {
            "name": "赛事",
            "type": "string",
            "description": "联赛或赛事的名称。"
        },
        {
            "name": "时间",
            "type": "string",
            "description": "比赛时间，格式为 'MM-DD HH:MM'。"
        },
        {
            "name": "主队",
            "type": "string",
            "description": "主队的名称。"
        },
        {
            "name": "客队",
            "type": "string",
            "description": "客队的名称。"
        },
        {
            "name": "主胜SP值",
            "type": "number",
            "description": "博彩公司为一场比赛中“主队获胜”开出的赔率。赔率越低，通常意味着该结果发生的概率被认为越高。"
        },
        {
            "name": "主平SP值",
            "type": "number",
            "description": "博彩公司为一场比赛“双方打平”开出的赔率。赔率同样反映了博彩公司认为平局发生的可能性。"
        },
        {
            "name": "主负SP值",
            "type": "number",
            "description": "博彩公司为一场比赛中“客队获胜”（从主队角度看就是主队负）开出的赔率。赔率越低，客队获胜的可能性被认为越高。"
        },
        {
            "name": "主胜概率",
            "type": "number",
            "description": "主队获胜的实际隐含概率，已去除庄家利润并归一化，更接近“公平”概率。"
        },
        {
            "name": "主平概率",
            "type": "number",
            "description": "比赛打平的实际隐含概率，已去除庄家利润并归一化。"
        },
        {
            "name": "主负概率",
            "type": "number",
            "description": "客队获胜（主队失利）的实际隐含概率，已去除庄家利润并归一化。"
        },
        {
            "name": "庄家抽水",
            "type": "number",
            "description": "博彩公司在赔率中设定的利润空间，即各结果隐含概率之和超出100%的部分。庄家抽水越高，对投注者越不利。"
        },
        {
            "name": "比赛结果",
            "type": "string",
            "description": "从主队视角看，比赛的实际结果（'胜'、'平'、'负'）。"
        }
        ],
    }
    }
    """
    
    DEFAULT_OUTPUT_COLS = [
        '期数id', '时间', '比赛id', '赛事', '主队', '客队',
        '主胜SP值', '主平SP值', '主负SP值',
        '主胜概率', '主平概率', '主负概率',
        '庄家抽水', '比赛结果'
    ]
    
    def __init__(
        self, 
        data_dir: str, 
        issue_threshold: int = 25000,
        output_cols: Optional[List[str]] = None
    ):
        """
        初始化处理器
        
        Args:
            data_dir: CSV文件所在目录
            issue_threshold: 期数id筛选阈值（默认25000）
            output_cols: 输出列名列表（默认使用预设列）
        """
        self.data_dir = Path(data_dir)
        self.issue_threshold = issue_threshold
        self.output_cols = output_cols or self.DEFAULT_OUTPUT_COLS
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> 'MatchOverview':
        """加载并合并所有CSV文件"""
        if not self.data_dir.exists():
            print(f"警告: 目录 {self.data_dir} 不存在。")
            self.raw_data = pd.DataFrame()
            return self

        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            print(f"警告: 目录 {self.data_dir} 中未找到CSV文件")
            self.raw_data = pd.DataFrame()
            return self
            
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file.name} 失败: {e}")
                
        if not dfs:
            self.raw_data = pd.DataFrame()
        else:
            self.raw_data = pd.concat(dfs, ignore_index=True)
        return self
    
    def _parse_numeric(self, cols: List[str]) -> None:
        """将指定列转换为数值类型"""
        if self.raw_data.empty:
            return
        for col in cols:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
    
    def _calculate_result(self, row: pd.Series) -> str:
        """根据比分判断比赛结果（主队视角）"""
        try:
            home, away = row['全场主队得分'], row['全场客队得分']
            if pd.isna(home) or pd.isna(away):
                return '未知'
            if home > away:
                return '胜'
            elif home == away:
                return '平'
            return '负'
        except:
            return '未知'
    
    def _get_value_by_result(self, row: pd.Series, value_map: Dict[str, str]) -> float:
        """根据比赛结果获取对应值（赔率或概率）"""
        if row['比赛结果'] == '未知':
            return np.nan
        return row.get(value_map.get(row['比赛结果'], ''), np.nan)
    
    def process(self) -> 'MatchOverview':
        """执行核心计算：结果判定、概率转换、赔率提取"""
        if self.raw_data is None:
            raise ValueError("请先调用load_data()加载数据")
            
        if self.raw_data.empty:
            self.processed_data = pd.DataFrame()
            return self

        # 确保数值类型
        score_cols = ['全场主队得分', '全场客队得分']
        odds_cols = ['主胜SP值', '主平SP值', '主负SP值']
        self._parse_numeric(score_cols + odds_cols)
        
        # 过滤掉缺少关键列的行
        required_cols = score_cols + odds_cols
        self.raw_data = self.raw_data.dropna(subset=odds_cols)
        
        if self.raw_data.empty:
            self.processed_data = pd.DataFrame()
            return self

        # 计算比赛结果
        self.raw_data['比赛结果'] = self.raw_data.apply(self._calculate_result, axis=1)
        
        # 计算隐含概率和庄家抽水
        # 避免除以零
        odds_values = self.raw_data[odds_cols].values
        odds_values[odds_values == 0] = np.nan
        
        probs = 1 / odds_values
        total_prob = np.nansum(probs, axis=1)
        
        # 避免 total_prob 为 0
        total_prob[total_prob == 0] = np.nan
        
        self.raw_data['主胜概率'] = probs[:, 0] / total_prob
        self.raw_data['主平概率'] = probs[:, 1] / total_prob
        self.raw_data['主负概率'] = probs[:, 2] / total_prob
        self.raw_data['庄家抽水'] = total_prob - 1
        
        # 提取实际发生的赔率和概率
        odds_map = {'胜': '主胜SP值', '平': '主平SP值', '负': '主负SP值'}
        prob_map = {'胜': '主胜概率', '平': '主平概率', '负': '主负概率'}
        
        self.raw_data['最终赔率'] = self.raw_data.apply(
            lambda row: self._get_value_by_result(row, odds_map), axis=1
        )
        self.raw_data['最终概率'] = self.raw_data.apply(
            lambda row: self._get_value_by_result(row, prob_map), axis=1
        )
        
        self.processed_data = self.raw_data.copy()
        return self
    
    def get_output(self, round_decimals: int = 3) -> pd.DataFrame:
        """
        获取处理后的输出数据（默认筛选期数id大于阈值的数据）
        
        Args:
            round_decimals: 数值保留小数位数（默认3位）
            
        Returns:
            处理后的DataFrame
        """
        if self.processed_data is None:
            raise ValueError("请先调用process()处理数据")
            
        if self.processed_data.empty:
            return pd.DataFrame(columns=self.output_cols)

        # 确保期数id列存在
        if '期数id' not in self.processed_data.columns:
            return pd.DataFrame(columns=self.output_cols)

        mask = self.processed_data['期数id'] > self.issue_threshold
        
        # 确保所有输出列都存在，不存在的填充为NaN
        for col in self.output_cols:
            if col not in self.processed_data.columns:
                self.processed_data[col] = np.nan

        df_out = self.processed_data.loc[mask, self.output_cols].copy()
        
        # 对数值列进行四舍五入
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        df_out[numeric_cols] = df_out[numeric_cols].round(round_decimals)
        
        return df_out
    
    def get_league_margin_stats(self) -> pd.Series:
        """获取各赛事的平均庄家抽水比例"""
        if self.processed_data is None:
            raise ValueError("请先调用process()处理数据")
        return self.processed_data.groupby('赛事')['庄家抽水'].mean().round(4)
    
    def run(self) -> pd.DataFrame:
        """一键执行完整流程"""
        return self.load_data().process().get_output()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 初始化处理器（可自定义路径、阈值、输出列）
    processor = MatchOverview(
        data_dir='D:\\MyProject\\football-data-mining\\data\\overview',
        issue_threshold=25000
    )
    
    # 方式1：链式调用
    df_output = processor.load_data().process().get_output()
    
    # 方式2：一键执行
    # df_output = processor.run()
    
    # 查看结果
    print(f"输出数据形状: {df_output.shape}")
    print(df_output.head())
    
    # 获取各赛事抽水统计
    margin_stats = processor.get_league_margin_stats()
    print("\n各赛事平均抽水比例:")
    print(margin_stats)