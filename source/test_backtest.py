from data import DataSource
from engine import ProbabilityEngine
from backtest import Ren9Backtest

def main():
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtest = Ren9Backtest(ds, pe)
    
    period_ids = [25192, 25193]
    
    print(f"=== 任选9回测测试 ===")
    print(f"测试期数: {period_ids}")
    print(f"成本上限: 128元")
    
    results = backtest.run_backtest(period_ids, max_cost=128, risk_tolerance=0.5)
    
    print("\n=== 回测结果 ===")
    print(results)
    
    print("\n=== 核心指标 ===")
    metrics = backtest.calculate_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
