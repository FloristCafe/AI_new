# Optiver Realized Volatility Starter

当前阶段目标不是立刻建模，而是先完成三件事：

- 看懂 `train.csv`、`book_train.parquet`、`trade_train.parquet` 的结构
- 从全量高频数据中抽出一个小而完整的联动沙盒样本
- 把 `book/trade` 明细聚合成第一版样本级 baseline 特征

当前脚本：

- `src/inspect_optiver_data.py`
- `src/build_optiver_sandbox.py`
- `src/build_optiver_baseline_features.py`
- `src/build_optiver_features_v2.py`
- `src/build_optiver_knn_features.py`
- `src/train_optiver_baseline.py`

建议运行顺序：

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\inspect_optiver_data.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\build_optiver_sandbox.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\build_optiver_baseline_features.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\build_optiver_features_v2.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\build_optiver_knn_features.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\train_optiver_baseline.py"
```
