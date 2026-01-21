import pandas as pd
import numpy as np
import itertools
from itertools import combinations
import warnings

data_final_prob = {
    '赛事': ['英超', '英冠', '非洲杯', '亚冠', '英联杯', '西甲'],
    'Avg_最终概率_平': [0.2484, 0.2687, 0.2868, 0.2522, 0.2238, 0.2688],
    'Avg_最终概率_胜': [0.5402, 0.459, 0.5037, 0.5596, 0.5033, 0.5334],
    'Avg_最终概率_负': [0.4355, 0.358, 0.3535, 0.486, 0.4541, 0.4077],
    'Std_最终概率_平': [0.0356, 0.0356, 0.0416, 0.0393, 0.0064, 0.0459],
    'Std_最终概率_胜': [0.1772, 0.1339, 0.1699, 0.1867, 0.1337, 0.1683],
    'Std_最终概率_负': [0.1725, 0.1298, 0.2506, 0.1435, 0.1724, 0.1661]
}
df_final_prob = pd.DataFrame(data_final_prob)

data_bonus = {
    '期号': ['25189', '25190', '25191', '25192', '25193'],
    '赛果冷热统计': ['比较冷', '一般', '一般', '一般', '超级冷'],
    '任九销量（元）': [15010990, 28872256, 23180498, 13490720, 15612194],
    '任九一等奖': [11210, 1906, 466, 195, 79934],
    '开奖时间': ['2025-12-19 00:00:00', '2025-12-21 00:00:00', '2025-12-22 00:00:00', '2025-12-25 00:00:00', '2025-12-27 00:00:00']
}
df_bonus = pd.DataFrame(data_bonus)

data_matches = {
"期数id":[25193,25193,25193,25193,25193,25193,25193,25193,25193,25193,25193,25193,25193,25193,25192,25192,25192,25192,25192,25192,25192,25192,25192,25192,25192,25192,25192,25192],"赛事":["英超","英冠","英冠","英冠","英冠","英冠","英冠","英冠","英冠","英冠","英冠","非洲杯","非洲杯","非洲杯","亚冠","亚冠","亚冠","亚冠","英超","英联杯","西甲","非洲杯","非洲杯","非洲杯","非洲杯","非洲杯","非洲杯","非洲杯"],
"主队":["曼联","米尔沃尔","考文垂","莱切城","米堡","诺维奇","牛津联","朴次茅斯","斯托克城","西布朗","雷克斯汉姆","埃及","赞比亚","摩洛哥","沙瑞加","巴格达警察","多哈萨德","吉达联合","富勒姆","阿森纳","毕尔巴鄂","马里","南非","埃及","塞内加尔","布基纳法索","科特迪瓦","喀麦隆"],
"客队":["纽卡斯尔","伊普斯","斯旺西","沃特福德","布莱克本","查尔顿","南安普敦","女王巡游","普雷斯顿","布里斯托","谢菲联","南非","科摩罗","马里","利雅新月","吉达国民","迪拜青年国民","纳萨夫","诺丁汉森林","水晶宫","西班牙人","赞比亚","安哥拉","津巴布韦","博茨瓦纳","赤道几内亚","莫桑比克","加蓬"],
"主胜SP值":[2.49,3.82,1.55,2.24,1.71,2.03,3.82,3.03,2.02,2.05,3.03,1.99,2.28,1.47,5.58,3.92,2.39,1.16,2.55,1.49,1.85,1.73,2.76,1.32,1.11,1.87,1.28,1.98],
"主平SP值":[3.64,3.3,4.18,3.41,3.78,3.46,3.75,3.32,3.25,3.37,3.4,3.03,2.93,3.92,4.3,3.7,3.5,7.06,3.31,4.3,3.34,3.31,2.64,4.55,8.38,2.97,5.01,3.05],
"主负SP值":[2.66,1.98,5.41,3.04,4.58,3.48,1.85,2.29,3.76,3.51,2.25,4.23,3.44,7.45,1.48,1.78,2.62,12.07,2.79,6.42,4.63,5.32,3.04,11.04,22.92,5.01,11.7,4.21],
"主胜概率":[0.382,0.245,0.603,0.418,0.548,0.461,0.245,0.309,0.463,0.456,0.309,0.47,0.41,0.636,0.165,0.235,0.385,0.793,0.373,0.633,0.512,0.541,0.339,0.709,0.847,0.499,0.733,0.472],
"主平概率":[0.261,0.283,0.224,0.274,0.248,0.27,0.249,0.282,0.288,0.277,0.275,0.309,0.319,0.239,0.214,0.249,0.263,0.13,0.287,0.22,0.284,0.283,0.354,0.206,0.112,0.314,0.187,0.306],
"主负概率":[0.357,0.472,0.173,0.308,0.204,0.269,0.506,0.409,0.249,0.266,0.416,0.221,0.272,0.125,0.621,0.517,0.352,0.076,0.34,0.147,0.205,0.176,0.307,0.085,0.041,0.186,0.08,0.222],
"庄家抽水":[0.052,0.07,0.069,0.069,0.068,0.069,0.069,0.068,0.069,0.069,0.069,0.069,0.071,0.07,0.087,0.087,0.086,0.087,0.053,0.059,0.056,0.068,0.07,0.068,0.064,0.071,0.066,0.07],
"比赛结果":["胜","平","胜","负","平","胜","胜","平","平","负","胜","胜","平","平","负","负","胜","胜","胜","平","负","平","胜","胜","胜","胜","胜","胜"]
}
df_matches = pd.DataFrame(data_matches)

data_outcome_freq = {
    '任九赛果冷热': ['一般', '比较冷', '超级冷'],
    '胜': [6.84, 5.66, 5.42],
    '平': [3.14, 3.9, 4.29],
    '负': [4.2, 4.6, 4.34]
}
df_outcome_freq = pd.DataFrame(data_outcome_freq)


data_outcome_prob = {
    '任九赛果冷热': ['一般', '比较冷', '超级冷'],
    '平': [0.26, 0.254, 0.252],
    '胜': [0.56, 0.513, 0.465],
    '负': [0.452, 0.393, 0.351]
}
df_outcome_prob = pd.DataFrame(data_outcome_prob)


data_leagues = {
    '赛事': ['英超', '英冠', '非洲杯', '亚冠', '英联杯', '西甲'],
    '样本量': [580.0, 247.0, 50.0, 71.0, 17.0, 438.0],
    '理论概率_胜': [0.442, 0.423, 0.441, 0.483, 0.449, 0.441],
    '理论概率_平': [0.235, 0.266, 0.277, 0.224, 0.241, 0.257],
    '理论概率_负': [0.323, 0.31, 0.282, 0.293, 0.31, 0.302],
    '真实频率_胜': [0.436, 0.453, 0.48, 0.634, 0.588, 0.457],
    '真实频率_平': [0.248, 0.231, 0.32, 0.183, 0.176, 0.251],
    '真实频率_负': [0.316, 0.316, 0.2, 0.183, 0.235, 0.292],
    '卡方统计量': [0.599, 1.718, 1.696, 6.78, 1.332, 0.422],
    'P值': [0.741, 0.424, 0.428, 0.034, 0.514, 0.81]
}
df_leagues = pd.DataFrame(data_leagues)


# 修复后的 Ren9QuantSystem 类
class Ren9QuantSystem:
    def __init__(self, df_matches, df_leagues, df_bonus, df_final_prob, df_outcome_freq, df_outcome_prob):
        self.df_matches = df_matches
        self.df_leagues = df_leagues
        self.df_bonus = df_bonus
        self.df_final_prob = df_final_prob
        self.df_outcome_freq = df_outcome_freq
        self.df_outcome_prob = df_outcome_prob
        self.initialize_system()
    
    def initialize_system(self):
        """初始化系统参数"""
        # 定义冷热等级阈值
        self.bonus_thresholds = {
            '火锅奖': 1000,
            '一般': 10000,
            '高奖金': 10000
        }
        
        # 联赛权重（基于卡方检验P值）
        self.league_weights = self.calculate_league_weights()
        
        # 奖金周期分析
        self.bonus_cycle = self.analyze_bonus_cycle()
    
    def calculate_league_weights(self):
        """
        优化后的联赛权重计算（科学重构）：
        1. 使用连续的 P 值评估，避免硬分界
        2. 使用根号样本量计算置信度
        3. 引入实际偏差强度 (MAE)
        """
        weights = {}
        for idx, row in self.df_leagues.iterrows():
            # A. 显著性得分：P值越小，潜力越大。使用 -log10 处理，并限制范围
            p_val = max(row['P值'], 0.001) # 避免 log(0)
            significance_score = -np.log10(p_val)
            p_weight = np.clip(significance_score / 1.3, 1.0, 3.0) # 1.3 是约 0.05 的 log 值
            
            # B. 置信度得分：使用根号样本量，100场为基准1.0，400场约等于2.0
            confidence_weight = np.sqrt(row['样本量'] / 100)
            confidence_weight = np.clip(confidence_weight, 0.5, 2.5)
            
            # C. 偏差强度：计算理论与真实的平均绝对偏差 (MAE)
            mae = (abs(row['理论概率_胜'] - row['真实频率_胜']) + 
                   abs(row['理论概率_平'] - row['真实频率_平']) + 
                   abs(row['理论概率_负'] - row['真实频率_负'])) / 3
            deviation_bonus = 1 + (mae * 5) # 偏差越大，额外奖励越高
            
            # 综合权重
            weights[row['赛事']] = round(p_weight * confidence_weight * deviation_bonus, 2)
            
        return weights
    
    def analyze_bonus_cycle(self):
        """分析奖金周期"""
        df = self.df_bonus.copy()
        df['火锅奖标识'] = df['任九一等奖'] < self.bonus_thresholds['火锅奖']
        
        # 计算连续火锅奖期数
        consecutive_counts = []
        current_count = 0
        for is_low in df['火锅奖标识'].values:
            if is_low:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        # 计算平均连续火锅奖期数
        avg_consecutive = np.mean(consecutive_counts) if consecutive_counts else 0
        
        return {
            '当前期号': df['期号'].iloc[-1],
            '当前奖金': df['任九一等奖'].iloc[-1],
            '平均连续火锅奖期数': avg_consecutive,
            '火锅奖次数': df['火锅奖标识'].sum(),
            '高奖金概率': 1 / (avg_consecutive + 1) if avg_consecutive > 0 else 0.3
        }
    
    def calculate_match_risk_score(self, period_id=None):
        """计算比赛风险评分"""
        if period_id:
            matches = self.df_matches[self.df_matches['期数id'] == period_id]
        else:
            matches = self.df_matches
        
        risk_scores = []
        
        for idx, row in matches.iterrows():
            league = row['赛事']
            
            # 获取联赛概率标准差
            league_stats = self.df_final_prob[self.df_final_prob['赛事'] == league]
            if len(league_stats) == 0:
                continue
            
            # 计算波动性风险
            prob_std = {
                '胜': league_stats['Std_最终概率_胜'].values[0],
                '平': league_stats['Std_最终概率_平'].values[0],
                '负': league_stats['Std_最终概率_负'].values[0]
            }
            
            # 计算当前比赛的概率分布熵（不确定性）
            probs = [row['主胜概率'], row['主平概率'], row['主负概率']]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            
            # 计算风险评分
            avg_std = np.mean(list(prob_std.values()))
            risk_score = entropy * avg_std * 10
            
            risk_scores.append({
                '期数': row['期数id'],
                '赛事': league,
                '主队': row['主队'],
                '客队': row['客队'],
                '风险评分': risk_score,
                '不确定性': entropy,
                '联赛波动性': avg_std
            })
        
        return pd.DataFrame(risk_scores)
    
    def identify_value_bets(self, period_id=None):
        """识别价值投注机会"""
        if period_id:
            matches = self.df_matches[self.df_matches['期数id'] == period_id]
        else:
            matches = self.df_matches
        
        value_bets = []
        
        for idx, row in matches.iterrows():
            # 获取联赛信息
            league = row['赛事']
            league_info = self.df_leagues[self.df_leagues['赛事'] == league]
            
            if len(league_info) == 0:
                continue
            
            # 计算理论概率与实际概率的偏差
            actual_prob = {
                '胜': row['主胜概率'],
                '平': row['主平概率'],
                '负': row['主负概率']
            }
            
            # 获取联赛平均理论概率
            theoretical_prob = {
                '胜': league_info['理论概率_胜'].values[0],
                '平': league_info['理论概率_平'].values[0],
                '负': league_info['理论概率_负'].values[0]
            }
            
            # 计算偏差值
            deviations = {}
            for outcome in ['胜', '平', '负']:
                deviation = actual_prob[outcome] - theoretical_prob[outcome]
                # 考虑联赛权重
                weight = self.league_weights.get(league, 1.0)
                deviations[outcome] = deviation * weight
            
            # 识别最大正偏差（价值投注机会）
            for outcome, deviation in deviations.items():
                if deviation > 0.05:  # 偏差阈值
                    value_bets.append({
                        '期数': row['期数id'],
                        '赛事': league,
                        '主队': row['主队'],
                        '客队': row['客队'],
                        '结果': outcome,
                        '实际概率': actual_prob[outcome],
                        '理论概率': theoretical_prob[outcome],
                        '偏差': deviation,
                        '赔率': row[f'主{outcome}SP值'],
                        '价值分数': deviation * actual_prob[outcome] * 100
                    })
        
        return pd.DataFrame(value_bets)
    
    def generate_betting_combinations(self, period_id, budget=100):
        """生成优化的投注组合"""
        # 获取当期比赛
        current_matches = self.df_matches[self.df_matches['期数id'] == period_id]
        
        # 识别价值投注
        value_bets = self.identify_value_bets(period_id)
        
        # 计算风险评分
        risk_scores = self.calculate_match_risk_score(period_id)
        
        # 合并数据
        matches_info = []
        for idx, row in current_matches.iterrows():
            # 查找价值投注信息
            value_info = value_bets[
                (value_bets['主队'] == row['主队']) & 
                (value_bets['客队'] == row['客队'])
            ]
            
            # 查找风险评分
            risk_info = risk_scores[
                (risk_scores['主队'] == row['主队']) & 
                (risk_scores['客队'] == row['客队'])
            ]
            
            value_score = value_info['价值分数'].values[0] if len(value_info) > 0 else 0
            risk_score = risk_info['风险评分'].values[0] if len(risk_info) > 0 else 1.0
            
            # 计算综合评分
            league_weight = self.league_weights.get(row['赛事'], 1.0)
            composite_score = value_score * league_weight / max(risk_score, 0.1)
            
            matches_info.append({
                '比赛ID': idx,
                '赛事': row['赛事'],
                '主队': row['主队'],
                '客队': row['客队'],
                '胜概率': row['主胜概率'],
                '平概率': row['主平概率'],
                '负概率': row['主负概率'],
                '胜赔率': row['主胜SP值'],
                '平赔率': row['主平SP值'],
                '负赔率': row['主负SP值'],
                '价值分数': value_score,
                '风险评分': risk_score,
                '综合评分': composite_score
            })
        
        df_matches_info = pd.DataFrame(matches_info)
        
        # 根据综合评分排序
        df_matches_info = df_matches_info.sort_values('综合评分', ascending=False)
        
        # 生成投注策略
        strategies = self.generate_strategies(df_matches_info, budget)
        
        return df_matches_info, strategies
    
    def generate_strategies(self, df_matches_info, budget):
        """生成具体投注策略"""
        strategies = []
        
        # 策略1: 高价值组合（激进型）
        high_value_matches = df_matches_info.head(12)  # 取前12场
        if len(high_value_matches) >= 9:
            strategies.append({
                '策略名称': '高价值激进型',
                '投注场次': 9,
                '胆码数量': 3,
                '拖码数量': 6,
                '预计成本': 2 * 56,  # 3胆6拖
                '预期收益倍数': self.estimate_return(high_value_matches.head(9), '激进'),
                '推荐联赛': high_value_matches['赛事'].unique().tolist()
            })
        
        # 策略2: 平衡组合（稳健型）
        balanced_matches = df_matches_info.iloc[3:12]  # 取中间9场
        if len(balanced_matches) >= 9:
            strategies.append({
                '策略名称': '平衡稳健型',
                '投注场次': 9,
                '胆码数量': 2,
                '拖码数量': 7,
                '预计成本': 2 * 42,  # 2胆7拖
                '预期收益倍数': self.estimate_return(balanced_matches.head(9), '稳健'),
                '推荐联赛': balanced_matches['赛事'].unique().tolist()
            })
        
        # 策略3: 冷门捕捉型（博大奖）
        # 选择风险评分高的比赛
        risky_matches = df_matches_info.sort_values('风险评分', ascending=False).head(10)
        if len(risky_matches) >= 9:
            strategies.append({
                '策略名称': '冷门捕捉型',
                '投注场次': 9,
                '胆码数量': 1,
                '拖码数量': 8,
                '预计成本': 2 * 72,  # 1胆8拖
                '预期收益倍数': self.estimate_return(risky_matches.head(9), '激进'),
                '推荐联赛': risky_matches['赛事'].unique().tolist(),
                '适用期数': '连续火锅奖后'
            })
        
        return strategies
    
    def estimate_return(self, matches_df, strategy_type):
        """估算预期收益倍数"""
        # 计算平均综合评分
        avg_score = matches_df['综合评分'].mean()
        
        # 根据策略类型调整
        if strategy_type == '激进':
            multiplier = 2.0
        elif strategy_type == '稳健':
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        # 考虑奖金周期
        cycle_factor = self.bonus_cycle['高奖金概率']
        
        # 估算收益倍数
        estimated_return = avg_score * multiplier * cycle_factor / 10
        
        return min(max(estimated_return, 0.5), 5.0)  # 限制在0.5-5倍之间
    
    def get_betting_recommendation(self, period_id=None):
        """获取投注推荐"""
        if not period_id:
            period_id = self.df_matches['期数id'].max()
        
        # 分析奖金周期
        cycle_info = self.bonus_cycle
        current_bonus = self.df_bonus[self.df_bonus['期号'] == str(period_id)]
        if len(current_bonus) > 0:
            current_bonus_value = current_bonus['任九一等奖'].values[0]
        else:
            current_bonus_value = None
        
        # 生成投注组合
        matches_info, strategies = self.generate_betting_combinations(period_id)
        
        # 价值投注识别
        value_bets = self.identify_value_bets(period_id)
        
        # 风险评估
        risk_scores = self.calculate_match_risk_score(period_id)
        
        return {
            '分析期数': period_id,
            '奖金周期分析': cycle_info,
            '当期奖金': current_bonus_value,
            '投注建议': self.get_betting_advice(cycle_info, current_bonus_value),
            '价值投注机会': value_bets.to_dict('records') if len(value_bets) > 0 else [],
            '高风险比赛': risk_scores.sort_values('风险评分', ascending=False).head(5).to_dict('records'),
            '投注策略推荐': strategies,
            '综合评分前十比赛': matches_info.head(10).to_dict('records')
        }
    
    def get_betting_advice(self, cycle_info, current_bonus):
        """根据奖金周期给出投注建议"""
        advice = []
        
        # 连续火锅奖后的策略
        if current_bonus and current_bonus < self.bonus_thresholds['火锅奖']:
            # 检查是否连续火锅奖
            recent_bonuses = self.df_bonus.tail(3)['任九一等奖'].values
            if all(b < self.bonus_thresholds['火锅奖'] for b in recent_bonuses):
                advice.append({
                    '类型': '周期机会',
                    '建议': '连续火锅奖后，高奖金概率大增，建议增加投注额度',
                    '推荐额度': '增加50%-100%',
                    '策略': '采用冷门捕捉型策略'
                })
        
        # 高奖金后的策略
        if current_bonus and current_bonus > self.bonus_thresholds['高奖金']:
            advice.append({
                '类型': '风险提示',
                '建议': '刚出现高奖金，下期可能回归火锅奖，建议保守投注',
                '推荐额度': '减少50%',
                '策略': '采用平衡稳健型策略'
            })
        
        # 一般情况
        if not advice:
            advice.append({
                '类型': '常规投注',
                '建议': '奖金周期正常，按常规策略投注',
                '推荐额度': '保持正常额度',
                '策略': '根据价值投注机会选择策略'
            })
        
        return advice

class ArbitrageOpportunityDetector:
    def __init__(self, df_matches, df_leagues):
        self.df_matches = df_matches
        self.df_leagues = df_leagues
    
    def find_probability_arbitrage(self, period_id=None):
        """寻找概率套利机会"""
        if period_id:
            matches = self.df_matches[self.df_matches['期数id'] == period_id]
        else:
            matches = self.df_matches
        
        arbitrage_opps = []
        
        for idx, row in matches.iterrows():
            # 获取联赛实际频率
            league = row['赛事']
            league_data = self.df_leagues[self.df_leagues['赛事'] == league]
            
            if len(league_data) == 0:
                continue
            
            # 计算每个结果的预期价值
            expected_values = {}
            for outcome in ['胜', '平', '负']:
                prob = row[f'主{outcome}概率']
                odds = row[f'主{outcome}SP值']
                expected_value = odds * prob - 1
                expected_values[outcome] = expected_value
            
            # 寻找正期望值的机会
            for outcome, ev in expected_values.items():
                if ev > -0.05:  # 期望值大于10%
                    # 检查是否为冷门
                    is_underdog = self.is_underdog(row, outcome)
                    
                    arbitrage_opps.append({
                        '期数': row['期数id'],
                        '赛事': league,
                        '主队': row['主队'],
                        '客队': row['客队'],
                        '投注选项': outcome,
                        '赔率': row[f'主{outcome}SP值'],
                        '概率': row[f'主{outcome}概率'],
                        '期望值': ev,
                        '是否为冷门': is_underdog,
                        '套利分数': ev * (2 if is_underdog else 1) * 100
                    })
        
        # 修复：检查列表是否为空，然后创建DataFrame
        if arbitrage_opps:
            return pd.DataFrame(arbitrage_opps).sort_values('期望值', ascending=False)
        else:
            # 返回一个包含所有列的空DataFrame
            return pd.DataFrame(columns=['期数', '赛事', '主队', '客队', '投注选项', 
                                       '赔率', '概率', '期望值', '是否为冷门', '套利分数'])
    
    def is_underdog(self, match_row, outcome):
        """判断是否为冷门选项"""
        probs = {
            '胜': match_row['主胜概率'],
            '平': match_row['主平概率'],
            '负': match_row['主负概率']
        }
        
        # 找出概率最高的选项
        max_prob_outcome = max(probs, key=probs.get)
        
        return outcome != max_prob_outcome and probs[outcome] < 0.4
    
    def find_league_efficiency_arbitrage(self):
        """寻找联赛效率套利机会"""
        arbitrage_opps = []
        
        for idx, row in self.df_leagues.iterrows():
            league = row['赛事']
            
            # 找出理论概率与实际频率偏差最大的结果
            deviations = {
                '胜': abs(row['理论概率_胜'] - row['真实频率_胜']),
                '平': abs(row['理论概率_平'] - row['真实频率_平']),
                '负': abs(row['理论概率_负'] - row['真实频率_负'])
            }
            
            max_dev_outcome = max(deviations, key=deviations.get)
            max_deviation = deviations[max_dev_outcome]
            
            if max_deviation > 0.1 and row['P值'] < 0.1:
                # 存在显著偏差，可能存在套利机会
                arbitrage_opps.append({
                    '赛事': league,
                    '偏差最大结果': max_dev_outcome,
                    '偏差值': max_deviation,
                    '理论概率': row[f'理论概率_{max_dev_outcome}'],
                    '实际频率': row[f'真实频率_{max_dev_outcome}'],
                    'P值': row['P值'],
                    '套利方向': '多投' if row[f'真实频率_{max_dev_outcome}'] > row[f'理论概率_{max_dev_outcome}'] else '少投',
                    '置信度': 1 - row['P值']
                })
        
        # 修复：检查列表是否为空
        if arbitrage_opps:
            return pd.DataFrame(arbitrage_opps).sort_values('偏差值', ascending=False)
        else:
            # 返回一个包含所有列的空DataFrame
            return pd.DataFrame(columns=['赛事', '偏差最大结果', '偏差值', '理论概率', 
                                       '实际频率', 'P值', '套利方向', '置信度'])
    
    def find_cross_period_arbitrage(self, window_size=5):
        """寻找跨期套利机会"""
        # 分析奖金周期模式
        df_bonus = pd.DataFrame(data_bonus)
        df_bonus['奖金变化率'] = df_bonus['任九一等奖'].pct_change()
        df_bonus['连续低奖金'] = df_bonus['任九一等奖'].rolling(window=3).apply(
            lambda x: all(v < 1000 for v in x), raw=True
        )
        
        # 识别模式
        patterns = []
        for i in range(len(df_bonus) - 1):
            if df_bonus['连续低奖金'].iloc[i]:
                next_bonus = df_bonus['任九一等奖'].iloc[i + 1]
                if next_bonus > 5000:
                    patterns.append({
                        '低奖金期': df_bonus['期号'].iloc[i],
                        '高奖金期': df_bonus['期号'].iloc[i + 1],
                        '奖金增长倍数': next_bonus / df_bonus['任九一等奖'].iloc[i]
                    })
        
        # 修复：检查列表是否为空
        if patterns:
            return pd.DataFrame(patterns)
        else:
            # 返回一个包含所有列的空DataFrame
            return pd.DataFrame(columns=['低奖金期', '高奖金期', '奖金增长倍数'])

class CompleteRen9ArbitrageSystem:
    def __init__(self, df_matches, df_leagues, df_bonus, df_final_prob, df_outcome_freq, df_outcome_prob):
        self.quant_system = Ren9QuantSystem(
            df_matches, df_leagues, df_bonus, df_final_prob, df_outcome_freq, df_outcome_prob
        )
        self.arbitrage_detector = ArbitrageOpportunityDetector(df_matches, df_leagues)
        
    def run_complete_analysis(self, period_id=None):
        """运行完整的套利分析"""
        if not period_id:
            period_id = df_matches['期数id'].max()
        
        print("=" * 80)
        print("任选9量化套利系统 - 完整分析报告")
        print("=" * 80)
        
        # 1. 基础分析
        print("\n1. 基础分析")
        print("-" * 40)
        base_analysis = self.quant_system.get_betting_recommendation(period_id)
        print(f"分析期数: {base_analysis['分析期数']}")
        if base_analysis['当期奖金']:
            print(f"当期奖金: {base_analysis['当期奖金']}元")
        else:
            print("当期奖金: 数据缺失")
        
        # 2. 套利机会识别
        print("\n2. 套利机会识别")
        print("-" * 40)
        
        # 2.1 概率套利
        prob_arb = self.arbitrage_detector.find_probability_arbitrage(period_id)
        if not prob_arb.empty and len(prob_arb) > 0:  # 修复：检查DataFrame是否为空
            print("概率套利机会（期望值>10%）：")
            for idx, row in prob_arb.head(5).iterrows():
                print(f"  {row['主队']} vs {row['客队']} - {row['投注选项']} "
                      f"(赔率:{row['赔率']:.2f}, 期望值:{row['期望值']:.2%})")
        else:
            print("未发现显著的概率套利机会")
        
        # 2.2 联赛效率套利
        league_arb = self.arbitrage_detector.find_league_efficiency_arbitrage()
        if not league_arb.empty and len(league_arb) > 0:  # 修复：检查DataFrame是否为空
            print("\n联赛效率套利机会：")
            for idx, row in league_arb.iterrows():
                print(f"  {row['赛事']}: {row['偏差最大结果']} "
                      f"(偏差:{row['偏差值']:.3f}, {row['套利方向']}, 置信度:{row['置信度']:.2%})")
        
        # 2.3 跨期套利
        cross_arb = self.arbitrage_detector.find_cross_period_arbitrage()
        if not cross_arb.empty and len(cross_arb) > 0:  # 修复：检查DataFrame是否为空
            print("\n跨期套利模式识别：")
            for idx, row in cross_arb.iterrows():
                print(f"  低奖金期{row['低奖金期']} → 高奖金期{row['高奖金期']} "
                      f"(增长{row['奖金增长倍数']:.1f}倍)")
        
        # 3. 投注策略推荐
        print("\n3. 投注策略推荐")
        print("-" * 40)
        if base_analysis['投注策略推荐']:
            for strategy in base_analysis['投注策略推荐']:
                print(f"{strategy['策略名称']}:")
                print(f"  投注方式: {strategy['胆码数量']}胆{strategy['拖码数量']}拖")
                print(f"  预计成本: {strategy['预计成本']}元")
                print(f"  预期收益倍数: {strategy['预期收益倍数']:.1f}倍")
                if '推荐联赛' in strategy:
                    print(f"  推荐联赛: {', '.join(strategy['推荐联赛'])}")
                print()
        else:
            print("未生成有效的投注策略")
        
        # 4. 风险提示
        print("\n4. 风险提示")
        print("-" * 40)
        if len(base_analysis['高风险比赛']) > 0:
            print("高风险比赛（谨慎选择）：")
            for match in base_analysis['高风险比赛'][:3]:
                print(f"  {match['主队']} vs {match['客队']} "
                      f"(风险评分:{match['风险评分']:.2f})")
        
        # 5. 资金管理建议
        print("\n5. 资金管理建议")
        print("-" * 40)
        if base_analysis['投注建议']:
            for advice in base_analysis['投注建议']:
                print(f"{advice['类型']}: {advice['建议']}")
                print(f"  推荐额度: {advice['推荐额度']}")
                print(f"  推荐策略: {advice['策略']}")
        
        # 返回完整结果
        return {
            '基础分析': base_analysis,
            '概率套利机会': prob_arb.to_dict('records') if not prob_arb.empty else [],
            '联赛效率套利': league_arb.to_dict('records') if not league_arb.empty else [],
            '跨期套利模式': cross_arb.to_dict('records') if not cross_arb.empty else []
        }
    
    def generate_optimized_bet_slip(self, period_id, budget=200):
        """生成优化投注单"""
        # 获取价值投注
        value_bets = self.quant_system.identify_value_bets(period_id)
        
        # 获取风险评分
        risk_scores = self.quant_system.calculate_match_risk_score(period_id)
        
        # 获取当期比赛
        current_matches = df_matches[df_matches['期数id'] == period_id]
        
        # 创建投注单
        bet_slip = {
            '期数': period_id,
            '预算': budget,
            '胆码': [],
            '双选': [],
            '全包': [],
            '预计成本': 0,
            '覆盖场次': 0
        }
        
        # 根据价值和风险筛选比赛
        for idx, row in current_matches.iterrows():
            # 查找价值信息
            value_info = value_bets[
                (value_bets['主队'] == row['主队']) & 
                (value_bets['客队'] == row['客队'])
            ]
            
            # 查找风险信息
            risk_info = risk_scores[
                (risk_scores['主队'] == row['主队']) & 
                (risk_scores['客队'] == row['客队'])
            ]
            
            if len(value_info) > 0 and len(risk_info) > 0:
                value_score = value_info['价值分数'].values[0] if '价值分数' in value_info.columns else 0
                risk_score = risk_info['风险评分'].values[0] if '风险评分' in risk_info.columns else 1.0
                
                # 分类投注类型
                if value_score > 50 and risk_score < 1.0:
                    # 高价值低风险 -> 胆码
                    recommended_outcome = value_info['结果'].values[0] if '结果' in value_info.columns else '胜'
                    bet_slip['胆码'].append({
                        '比赛': f"{row['主队']} vs {row['客队']}",
                        '选择': recommended_outcome,
                        '赔率': row[f'主{recommended_outcome}SP值'],
                        '概率': row[f'主{recommended_outcome}概率'],
                        '价值分数': value_score
                    })
                elif value_score > 30:
                    # 中等价值 -> 双选
                    if '结果' in value_info.columns:
                        recommended_outcome = value_info['结果'].values[0]
                    else:
                        # 选择最可能的一个结果和次可能的结果
                        probs = [(row['主胜概率'], '胜'), 
                                (row['主平概率'], '平'), 
                                (row['主负概率'], '负')]
                        probs.sort(reverse=True)
                        recommended_outcome = probs[0][1]
                    
                    # 选择最可能的一个结果和次可能的结果
                    probs = [(row['主胜概率'], '胜'), 
                            (row['主平概率'], '平'), 
                            (row['主负概率'], '负')]
                    probs.sort(reverse=True)
                    selections = [probs[0][1], probs[1][1]]
                    
                    bet_slip['双选'].append({
                        '比赛': f"{row['主队']} vs {row['客队']}",
                        '选择': selections,
                        '赔率': [row[f'主{s}SP值'] for s in selections],
                        '价值分数': value_score
                    })
                else:
                    # 低价值高风险 -> 全包或放弃
                    pass
        
        # 限制数量
        bet_slip['胆码'] = bet_slip['胆码'][:3]  # 最多3胆
        bet_slip['双选'] = bet_slip['双选'][:6]  # 最多6场双选
        
        # 计算预计成本
        num_dan = len(bet_slip['胆码'])
        num_shuang = len(bet_slip['双选'])
        total_matches = num_dan + num_shuang
        
        if total_matches >= 9:
            # 计算组合数
            if num_dan <= 3 and num_shuang >= 6:
                combinations = 2 ** num_shuang
                cost = combinations * 2
            else:
                # 需要补充全包场次
                need_matches = 9 - total_matches
                cost = 3 ** need_matches * 2 * (2 ** num_shuang)
        else:
            cost = 0
        
        bet_slip['预计成本'] = min(cost, budget)
        bet_slip['覆盖场次'] = total_matches
        
        return bet_slip

# 运行系统
if __name__ == "__main__":
    # 初始化系统，传入所有必需的数据
    system = CompleteRen9ArbitrageSystem(
        df_matches=df_matches,
        df_leagues=df_leagues,
        df_bonus=df_bonus,
        df_final_prob=df_final_prob,
        df_outcome_freq=df_outcome_freq,
        df_outcome_prob=df_outcome_prob
    )
    
    # 分析最新一期
    latest_period = df_matches['期数id'].max()
    
    # 运行完整分析
    print("正在运行任选9量化套利系统...")
    results = system.run_complete_analysis(latest_period)
    
    # 生成优化投注单
    print("\n" + "=" * 80)
    print("优化投注单生成")
    print("=" * 80)
    
    bet_slip = system.generate_optimized_bet_slip(latest_period, budget=200)
    
    print(f"\n期数: {bet_slip['期数']}")
    print(f"预算: {bet_slip['预算']}元")
    print(f"预计成本: {bet_slip['预计成本']}元")
    print(f"覆盖场次: {bet_slip['覆盖场次']}场")
    
    if bet_slip['胆码']:
        print("\n胆码推荐:")
        for bet in bet_slip['胆码']:
            print(f"  {bet['比赛']}: {bet['选择']} (赔率:{bet['赔率']:.2f}, "
                  f"概率:{bet['概率']:.1%}, 价值分:{bet['价值分数']:.1f})")
    else:
        print("\n无胆码推荐")
    
    if bet_slip['双选']:
        print("\n双选推荐:")
        for bet in bet_slip['双选']:
            print(f"  {bet['比赛']}: {bet['选择']} "
                  f"(赔率:{bet['赔率'][0]:.2f}/{bet['赔率'][1]:.2f}, "
                  f"价值分:{bet['价值分数']:.1f})")
    else:
        print("\n无双选推荐")
    
    # 保存结果
    try:
        import json
        # with open('ren9_arbitrage_results.json', 'w', encoding='utf-8') as f:
            # json.dump(results, f, ensure_ascii=False, indent=2)
        print(results)
        # print("\n分析结果已保存至 ren9_arbitrage_results.json")
    except Exception as e:
        print(f"\n保存结果时出错: {e}")
    
    print("\n系统分析完成！")