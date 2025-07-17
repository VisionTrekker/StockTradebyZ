#!/bin/bash

echo "开始拉取数据..."
set -x  # 开启调试模式，打印接下来要执行的命令
python fetch_kline.py --datasource tushare --frequency 4 --exclude-gem False --min-mktcap 5e9 --max-mktcap +inf --start 20220101 --end today --out ./data --workers 20
set +x  # 关闭调试模式

# 检查上一个命令是否执行成功
if [ $? -ne 0 ]; then
    echo "数据拉取失败，退出脚本。"
    exit 1
fi

set -x
current_hour=$(date +%H)  # 获取当前小时数
# 根据当前时间确定选股日期
if [ "$current_hour" -ge 15 ]; then
    # 晚上 15 点之后，选股日期为下一天
    select_date=$(date -d "tomorrow" +%Y-%m-%d)
else
    # 早上，选股日期为当天
    select_date=$(date +%Y-%m-%d)
fi
set +x

echo "开始选股..."
set -x
python select_stock.py --data-dir ./data --config ./configs.json --date "$select_date"
set +x

# 检查上一个命令是否执行成功
if [ $? -ne 0 ]; then
    echo "选股操作失败，退出脚本。"
    exit 1
fi

echo "数据拉取和选股操作均成功完成。"