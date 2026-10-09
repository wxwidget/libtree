# LibTree CPU 多线程实测

3 个线程数 × 7 个数据集 × 3 个划分 × C++/Python × 3 个引擎 × 3 次计时，共 **1134 次**。

Equal configured thread budgets; independent processes. Each cell uses the pooled median of 9 runs. All LibTree same-split quality metrics are unchanged across threads and from the preceding five-dataset snapshot.

训练与预测分开测量。每组 9 次原始计时，保留最小/最大值；效果为三个划分的均值及样本标准差。

可用逻辑 CPU：5；cgroup CPU 配额：400000 100000；种子：[42, 2024, 2026]。线程数为预算，小任务可串行。

| 数据集 | 接口 | 引擎 | 线程 | 训练 ms | 推理 ms | 训练加速 | 推理加速 | 指标 | 测试效果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| friedman_100k | cpp | libtree | 1 | 554.04 | 14.835 | 1.00× | 1.00× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | libtree | 2 | 446.31 | 11.437 | 1.24× | 1.30× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | libtree | 4 | 389.69 | 6.675 | 1.42× | 2.22× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | lightgbm | 1 | 599.46 | 87.040 | 1.00× | 1.00× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | lightgbm | 2 | 370.01 | 46.297 | 1.62× | 1.88× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | lightgbm | 4 | 290.52 | 25.126 | 2.06× | 3.46× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | xgboost | 1 | 539.46 | 23.990 | 1.00× | 1.00× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | cpp | xgboost | 2 | 490.32 | 22.413 | 1.10× | 1.07× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | cpp | xgboost | 4 | 533.81 | 23.358 | 1.01× | 1.03× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | libtree | 1 | 531.25 | 15.427 | 1.00× | 1.00× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | libtree | 2 | 428.85 | 10.100 | 1.24× | 1.53× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | libtree | 4 | 383.94 | 6.814 | 1.38× | 2.26× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | lightgbm | 1 | 669.44 | 92.730 | 1.00× | 1.00× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | lightgbm | 2 | 400.30 | 51.415 | 1.67× | 1.80× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | lightgbm | 4 | 296.37 | 27.659 | 2.26× | 3.35× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | xgboost | 1 | 529.38 | 20.585 | 1.00× | 1.00× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | xgboost | 2 | 508.68 | 19.740 | 1.04× | 1.04× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | xgboost | 4 | 539.39 | 20.147 | 0.98× | 1.02× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_20k | cpp | libtree | 1 | 107.18 | 3.026 | 1.00× | 1.00× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | libtree | 2 | 95.37 | 1.772 | 1.12× | 1.71× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | libtree | 4 | 89.40 | 1.389 | 1.20× | 2.18× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | lightgbm | 1 | 129.18 | 16.303 | 1.00× | 1.00× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | lightgbm | 2 | 85.14 | 9.032 | 1.52× | 1.80× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | lightgbm | 4 | 62.38 | 4.998 | 2.07× | 3.26× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | xgboost | 1 | 131.00 | 4.700 | 1.00× | 1.00× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | cpp | xgboost | 2 | 132.82 | 4.850 | 0.99× | 0.97× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | cpp | xgboost | 4 | 124.92 | 4.973 | 1.05× | 0.95× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | libtree | 1 | 101.39 | 2.961 | 1.00× | 1.00× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | libtree | 2 | 100.22 | 1.873 | 1.01× | 1.58× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | libtree | 4 | 90.48 | 1.521 | 1.12× | 1.95× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | lightgbm | 1 | 145.51 | 17.916 | 1.00× | 1.00× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | lightgbm | 2 | 103.35 | 11.117 | 1.41× | 1.61× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | lightgbm | 4 | 71.67 | 6.254 | 2.03× | 2.86× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | xgboost | 1 | 134.21 | 4.796 | 1.00× | 1.00× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | xgboost | 2 | 133.15 | 4.775 | 1.01× | 1.00× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | xgboost | 4 | 142.37 | 5.083 | 0.94× | 0.94× | RMSE ↓ | 1.1556 ± 0.0085 |
| insurance | cpp | libtree | 1 | 3.96 | 0.191 | 1.00× | 1.00× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | libtree | 2 | 4.84 | 0.181 | 0.82× | 1.05× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | libtree | 4 | 4.46 | 0.180 | 0.89× | 1.06× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | lightgbm | 1 | 7.30 | 0.875 | 1.00× | 1.00× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | lightgbm | 2 | 9.57 | 0.633 | 0.76× | 1.38× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | lightgbm | 4 | 12.23 | 0.359 | 0.60× | 2.44× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | xgboost | 1 | 9.16 | 0.384 | 1.00× | 1.00× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | cpp | xgboost | 2 | 10.08 | 0.397 | 0.91× | 0.97× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | cpp | xgboost | 4 | 9.59 | 0.402 | 0.96× | 0.96× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | libtree | 1 | 4.60 | 0.216 | 1.00× | 1.00× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | libtree | 2 | 4.85 | 0.213 | 0.95× | 1.01× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | libtree | 4 | 5.27 | 0.222 | 0.87× | 0.97× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | lightgbm | 1 | 21.69 | 1.956 | 1.00× | 1.00× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | lightgbm | 2 | 21.90 | 1.300 | 0.99× | 1.51× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | lightgbm | 4 | 22.53 | 1.091 | 0.96× | 1.79× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | xgboost | 1 | 15.20 | 1.200 | 1.00× | 1.00× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | xgboost | 2 | 15.77 | 1.266 | 0.96× | 0.95× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | xgboost | 4 | 15.74 | 1.169 | 0.97× | 1.03× | RMSE ↓ | 4631.3675 ± 115.5969 |
| pima | cpp | libtree | 1 | 5.11 | 0.166 | 1.00× | 1.00× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | cpp | libtree | 2 | 5.14 | 0.166 | 0.99× | 1.00× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | cpp | libtree | 4 | 5.23 | 0.165 | 0.98× | 1.01× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | cpp | lightgbm | 1 | 11.17 | 0.671 | 1.00× | 1.00× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | cpp | lightgbm | 2 | 9.67 | 0.414 | 1.16× | 1.62× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | cpp | lightgbm | 4 | 9.19 | 0.220 | 1.22× | 3.06× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | cpp | xgboost | 1 | 14.08 | 0.405 | 1.00× | 1.00× | AUC ↑ | 0.8047 ± 0.0338 |
| pima | cpp | xgboost | 2 | 14.43 | 0.366 | 0.98× | 1.11× | AUC ↑ | 0.8047 ± 0.0338 |
| pima | cpp | xgboost | 4 | 15.28 | 0.420 | 0.92× | 0.96× | AUC ↑ | 0.8047 ± 0.0338 |
| pima | python | libtree | 1 | 5.51 | 0.246 | 1.00× | 1.00× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | python | libtree | 2 | 5.54 | 0.234 | 0.99× | 1.05× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | python | libtree | 4 | 5.85 | 0.248 | 0.94× | 0.99× | AUC ↑ | 0.8074 ± 0.0390 |
| pima | python | lightgbm | 1 | 22.03 | 1.398 | 1.00× | 1.00× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | python | lightgbm | 2 | 20.57 | 1.115 | 1.07× | 1.25× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | python | lightgbm | 4 | 26.13 | 1.054 | 0.84× | 1.33× | AUC ↑ | 0.8030 ± 0.0426 |
| pima | python | xgboost | 1 | 19.91 | 1.243 | 1.00× | 1.00× | AUC ↑ | 0.8047 ± 0.0338 |
| pima | python | xgboost | 2 | 20.11 | 1.383 | 0.99× | 0.90× | AUC ↑ | 0.8047 ± 0.0338 |
| pima | python | xgboost | 4 | 20.49 | 1.288 | 0.97× | 0.97× | AUC ↑ | 0.8047 ± 0.0338 |
| telco | cpp | libtree | 1 | 47.36 | 1.001 | 1.00× | 1.00× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | cpp | libtree | 2 | 47.75 | 0.654 | 0.99× | 1.53× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | cpp | libtree | 4 | 54.58 | 0.567 | 0.87× | 1.76× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | cpp | lightgbm | 1 | 60.11 | 5.381 | 1.00× | 1.00× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | cpp | lightgbm | 2 | 43.72 | 3.076 | 1.38× | 1.75× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | cpp | lightgbm | 4 | 46.84 | 1.838 | 1.28× | 2.93× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | cpp | xgboost | 1 | 57.33 | 2.260 | 1.00× | 1.00× | AUC ↑ | 0.8465 ± 0.0019 |
| telco | cpp | xgboost | 2 | 53.05 | 2.391 | 1.08× | 0.95× | AUC ↑ | 0.8465 ± 0.0019 |
| telco | cpp | xgboost | 4 | 59.19 | 2.444 | 0.97× | 0.92× | AUC ↑ | 0.8465 ± 0.0019 |
| telco | python | libtree | 1 | 47.94 | 1.266 | 1.00× | 1.00× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | python | libtree | 2 | 46.04 | 0.934 | 1.04× | 1.36× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | python | libtree | 4 | 47.89 | 0.720 | 1.00× | 1.76× | AUC ↑ | 0.8465 ± 0.0015 |
| telco | python | lightgbm | 1 | 72.99 | 6.455 | 1.00× | 1.00× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | python | lightgbm | 2 | 58.62 | 4.426 | 1.25× | 1.46× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | python | lightgbm | 4 | 51.08 | 2.784 | 1.43× | 2.32× | AUC ↑ | 0.8444 ± 0.0002 |
| telco | python | xgboost | 1 | 63.30 | 2.685 | 1.00× | 1.00× | AUC ↑ | 0.8465 ± 0.0019 |
| telco | python | xgboost | 2 | 63.61 | 2.734 | 1.00× | 0.98× | AUC ↑ | 0.8465 ± 0.0019 |
| telco | python | xgboost | 4 | 61.69 | 2.552 | 1.03× | 1.05× | AUC ↑ | 0.8465 ± 0.0019 |
| titanic | cpp | libtree | 1 | 4.27 | 0.151 | 1.00× | 1.00× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | cpp | libtree | 2 | 3.85 | 0.154 | 1.11× | 0.98× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | cpp | libtree | 4 | 4.28 | 0.152 | 1.00× | 0.99× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | cpp | lightgbm | 1 | 7.34 | 0.642 | 1.00× | 1.00× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | cpp | lightgbm | 2 | 8.15 | 0.339 | 0.90× | 1.89× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | cpp | lightgbm | 4 | 20.94 | 0.300 | 0.35× | 2.14× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | cpp | xgboost | 1 | 10.23 | 0.429 | 1.00× | 1.00× | AUC ↑ | 0.8325 ± 0.0145 |
| titanic | cpp | xgboost | 2 | 8.96 | 0.379 | 1.14× | 1.13× | AUC ↑ | 0.8325 ± 0.0145 |
| titanic | cpp | xgboost | 4 | 9.56 | 0.472 | 1.07× | 0.91× | AUC ↑ | 0.8325 ± 0.0145 |
| titanic | python | libtree | 1 | 4.55 | 0.228 | 1.00× | 1.00× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | python | libtree | 2 | 4.48 | 0.237 | 1.01× | 0.96× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | python | libtree | 4 | 4.49 | 0.221 | 1.01× | 1.03× | AUC ↑ | 0.8393 ± 0.0144 |
| titanic | python | lightgbm | 1 | 18.04 | 1.231 | 1.00× | 1.00× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | python | lightgbm | 2 | 21.16 | 1.258 | 0.85× | 0.98× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | python | lightgbm | 4 | 24.09 | 1.070 | 0.75× | 1.15× | AUC ↑ | 0.8337 ± 0.0143 |
| titanic | python | xgboost | 1 | 15.14 | 1.196 | 1.00× | 1.00× | AUC ↑ | 0.8325 ± 0.0145 |
| titanic | python | xgboost | 2 | 15.62 | 1.225 | 0.97× | 0.98× | AUC ↑ | 0.8325 ± 0.0145 |
| titanic | python | xgboost | 4 | 14.92 | 1.153 | 1.01× | 1.04× | AUC ↑ | 0.8325 ± 0.0145 |
| wine | cpp | libtree | 1 | 7.35 | 0.184 | 1.00× | 1.00× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | cpp | libtree | 2 | 6.89 | 0.182 | 1.07× | 1.01× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | cpp | libtree | 4 | 7.13 | 0.182 | 1.03× | 1.01× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | cpp | lightgbm | 1 | 11.82 | 1.167 | 1.00× | 1.00× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | cpp | lightgbm | 2 | 10.77 | 0.605 | 1.10× | 1.93× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | cpp | lightgbm | 4 | 11.20 | 0.408 | 1.06× | 2.86× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | cpp | xgboost | 1 | 19.79 | 0.546 | 1.00× | 1.00× | RMSE ↓ | 0.6481 ± 0.0236 |
| wine | cpp | xgboost | 2 | 19.26 | 0.456 | 1.03× | 1.20× | RMSE ↓ | 0.6481 ± 0.0236 |
| wine | cpp | xgboost | 4 | 19.39 | 0.423 | 1.02× | 1.29× | RMSE ↓ | 0.6481 ± 0.0236 |
| wine | python | libtree | 1 | 7.60 | 0.217 | 1.00× | 1.00× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | python | libtree | 2 | 7.95 | 0.223 | 0.96× | 0.97× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | python | libtree | 4 | 7.73 | 0.224 | 0.98× | 0.97× | RMSE ↓ | 0.6375 ± 0.0116 |
| wine | python | lightgbm | 1 | 23.09 | 1.829 | 1.00× | 1.00× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | python | lightgbm | 2 | 20.32 | 1.262 | 1.14× | 1.45× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | python | lightgbm | 4 | 29.40 | 1.245 | 0.79× | 1.47× | RMSE ↓ | 0.6389 ± 0.0115 |
| wine | python | xgboost | 1 | 26.10 | 1.301 | 1.00× | 1.00× | RMSE ↓ | 0.6481 ± 0.0236 |
| wine | python | xgboost | 2 | 25.31 | 1.261 | 1.03× | 1.03× | RMSE ↓ | 0.6481 ± 0.0236 |
| wine | python | xgboost | 4 | 25.81 | 1.289 | 1.01× | 1.01× | RMSE ↓ | 0.6481 ± 0.0236 |

## 排名与限制

1 线程：LibTree 训练 12/14 组最快，推理 14/14 组最快。
2 线程：LibTree 训练 10/14 组最快，推理 14/14 组最快。
4 线程：LibTree 训练 9/14 组最快，推理 14/14 组最快。

加速比为同版本单线程中位数除以当前线程中位数，不是配对逐轮实验。共享云 CPU 有波动，不能将差异解释为普遍最优。小任务可能因启动和同步变慢，全部结果均保留。

主参数：100 棵树、深度 4、64 分箱、学习率 0.1、L2=1、最小增益 0。LibTree/LightGBM 最小叶样本 5；XGBoost 默认最小 Hessian 权重 1；LightGBM leaf-wise、16 叶。各引擎分箱、生长方式并不完全相同，无超参数搜索。

训练包含分箱/DMatrix/标签传递及 LibTree 冷线程池准备；排除数据下载、CSV、预处理、导入和进程启动。预测计时包含接口必要校验和转换，C++ XGBoost 包含测试 DMatrix 构建；RSS 为执行进程 VmHWM。

75/25 留出，二分类分层；训练集拟合编码，保留 NaN，红酒先去重。Kaggle 对应数据来自固定 SHA256 的公开镜像；Friedman 为合成压力测试。未覆盖百万行、稀疏、多分类、GPU、跨硬件任务。

5/5 CTest、31 Python、13 CLI、4 基准准备及 2 性能/所有权用例通过；4,701 项原生检查。ASan/UBSan/LSan 与 ThreadSanitizer 均通过。源码行覆盖 1116/1116=100%，分支 1116/1660=67.2%，函数 106/106；Python 148/148=100%。仅排除明确标记的系统耗尽、异常边界、竞态和编译器行映射续行。远程 GitHub Actions 是否通过需另行查看。

## 复现

```sh
. /workspace/libtree-venv/bin/activate
make all
PYTHONPATH=python python benchmarks/parallel.py --threads 1 2 4 --seeds 42 2024 2026 --repeats 3
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/parallel_report.py
make test
make sanitize
make thread-sanitize
make coverage
```

[交互 HTML](PARALLEL_REPORT.html) · [全部原始计时](parallel-results.json) · [聚合数据](parallel-summary.json) · [并行原理](../docs/design.zh-CN.md#确定性并行)
