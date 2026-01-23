from data import DataSource
from engine import ProbabilityEngine
from backtest import Ren9Backtest

def main():
    ds = DataSource()
    pe = ProbabilityEngine(ds)
    backtest = Ren9Backtest(ds, pe)
    
    period_ids = [25189, 25192, 25193]
    
    print(f"=== 任选9回测测试 ===")
    print(f"测试期数: {period_ids}")
    print(f"成本上限: 128元")
    
    backtest.run_backtest(period_ids, max_cost=128, risk_tolerance=0.5)
    
    # 打印详细报告并保存到文件
    backtest.print_report()
    backtest.save_report_to_file("RX9_Backtest_Report.md")

if __name__ == "__main__":
    main()
