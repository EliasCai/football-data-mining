import pandas as pd
import numpy as np
import os
from data import DataSource
from engine import ProbabilityEngine
from strategy import RX9Optimizer
from bet import ColdnessPredictor

def run_prediction(period_id: str):
    """
    对指定期数进行全策略预测并输出 Markdown 表格
    """
    # 1. 初始化
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    optimizer = RX9Optimizer()
    predictor = ColdnessPredictor(threshold=0.4)

    # 2. 获取数据
    df_merged = pe.get_merged_data()
    
    # 确保期号类型匹配 (CSV 读取后可能是 int)
    try:
        pid_int = int(period_id)
    except ValueError:
        pid_int = period_id

    df_period = df_merged[df_merged['期数id'] == pid_int].head(14).reset_index(drop=True)
    
    if df_period.empty:
        print(f"错误: 找不到期号 {period_id} 的比赛数据。")
        return

    # 3. 冷热预测
    print(f"\n{'='*20} 正在处理期号: {period_id} {'='*20}")
    pred_cold, prob_cold = predictor.predict_latest(period_id)

    # 4. 策略配置
    strategy_configs = {
        'strategy_01': {'i': 1, 'j': 3, 'k': 3, 'l': 2},
        'strategy_02': {'i': 1, 'j': 3, 'k': 4, 'l': 1},
        'strategy_03': {'i': 2, 'j': 3, 'k': 3, 'l': 1}
    }

    # 5. 生成各策略投注建议
    results_map = {}
    for s_name, params in strategy_configs.items():
        try:
            res = optimizer.generate_ticket(
                df_period, 
                i=params['i'], j=params['j'], k=params['k'], l=params['l'], 
                strategy_name=s_name
            )
            # all_matches 包含 14 场比赛的记录，每条记录包含 '推荐' 字段
            results_map[s_name] = res['all_matches']
        except Exception as e:
            print(f"策略 {s_name} 生成失败: {e}")
            results_map[s_name] = []

    # 6. 整合并打印 Markdown 表格
    table_rows = []
    for idx, row in df_period.iterrows():
        match_row = {
            '场次': idx + 1,
            '赛事': row['赛事'],
            '主队': row['主队'],
            '客队': row['客队'],
            '主胜SP': f"{row['主胜SP值']:.2f}",
            '主平SP': f"{row['主平SP值']:.2f}",
            '主负SP': f"{row['主负SP值']:.2f}"
        }
        
        # 添加各策略的推荐
        for s_name in strategy_configs.keys():
            matches = results_map.get(s_name, [])
            if idx < len(matches):
                bet = matches[idx].get('推荐', '')
                if not bet or bet == "":
                    bet = "-"
                match_row[s_name] = bet
            else:
                match_row[s_name] = "N/A"
        
        table_rows.append(match_row)

    df_display = pd.DataFrame(table_rows)
    
    print(f"\n### 期号 {period_id} 任选9 投注预测方案 (14场全表)")
    print(f"\n**冷热预测结果**: {'冷门' if pred_cold == 1 else '一般'} (概率: {prob_cold:.4f}, 阈值: {predictor.threshold})")
    print("\n" + df_display.to_markdown(index=False))

    # 保存结果到 CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"prediction_{period_id}.csv")
    df_display.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"\n[系统] 预测结果已保存至: {file_path}")

    # 7. 打印注数与成本
    cost_summary = []
    for s_name, params in strategy_configs.items():
        # 这里需要重新计算注数，或者在上面存储 res
        # 为了方便，我们这里直接打印汇总
        # 我们可以从 optimizer 重新调用一次或在上面保存 res 对象
        pass

    # 重新整理输出注数
    print("\n### 方案成本汇总")
    cost_rows = []
    for s_name, params in strategy_configs.items():
        try:
            # 重新生成一次以获取注数 (开销较小)
            res = optimizer.generate_ticket(df_period, **params, strategy_name=s_name)
            cost_rows.append({
                '策略': s_name,
                '参数': f"i={params['i']}, j={params['j']}, k={params['k']}, l={params['l']}",
                '总注数': res['total_notes'],
                '投注成本(元)': res['total_cost']
            })
        except:
            continue
    
    if cost_rows:
        print("\n" + pd.DataFrame(cost_rows).to_markdown(index=False))
if __name__ == "__main__":
    # 测试最新一期
    target_id = "26018"
    run_prediction(target_id)
