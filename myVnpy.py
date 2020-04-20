from vnpy.app.cta_strategy.csv_backtesting import CsvBacktestingEngine, OptimizationSetting
from vnpy.app.cta_strategy.base import BacktestingMode
from vnpy.app.cta_strategy.strategies.atr_rsi_strategy import (
    AtrRsiStrategy,
)
from datetime import datetime

engine = CsvBacktestingEngine()
engine.set_parameters(
    vt_symbol="IF88.CFFEX",
    interval="1m",
    start=datetime(2016, 1, 1),
    end=datetime(2019, 4, 30),
    rate=0.3/10000,
    slippage=0.2,
    size=300,
    pricetick=0.2,
    capital=1_000_000,
)
engine.add_strategy(AtrRsiStrategy, {})

engine.load_data("data.csv", names = [
    "datetime",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "open_interest",
])

engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()