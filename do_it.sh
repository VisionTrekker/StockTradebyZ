#!/bin/bash
set -e  # 遇到错误立即退出脚本
set -o pipefail  # 管道中的任意命令出错则返回错误


echo "开始拉取数据..."
set -x  # 开启调试模式，打印接下来要执行的命令
python fetch_kline.py \
    --datasource tushare \
    --frequency 4 \
    --exclude-gem \
    --min-mktcap 5e9 \
    --max-mktcap +inf \
    --start 20220101 \
    --end today \
    --out ./data \
    --workers 20
set +x  # 关闭调试模式

echo "数据拉取成功。准备选股..."

# 根据运行环境（本地或CI）确定时间判断阈值
if [ -n "$GITHUB_ACTIONS" ]; then
    # 在 GitHub Actions (UTC) 环境中运行时，阈值为 8 (对应北京时间 16:00)
    hour_threshold=8
    echo "Running in GitHub Actions (UTC). Using hour threshold: $hour_threshold"
else
    # 在本地 (假定为 UTC+8) 环境中运行时，阈值为 16
    hour_threshold=16
    echo "Running locally. Using hour threshold: $hour_threshold"
fi

current_hour=$(date +%H)  # 获取当前小时数
# 根据当前时间确定选股日期
if [ "$current_hour" -ge "$hour_threshold" ]; then
    # 收盘后，选股日期为下一天
    select_date=$(date -d "tomorrow" +%Y-%m-%d)
else
    # 收盘前，选股日期为当天
    select_date=$(date +%Y-%m-%d)
fi

echo "选股日期为：$select_date"
echo "开始选股..."
set -x
python select_stock.py \
    --data-dir ./data \
    --config ./configs.json \
    --date "$select_date"
set +x

echo "
✅ 数据拉取和选股操作均成功完成。"
echo "-----------------------------------------"
echo "执行邮件发送任务..."
python send_email.py
echo "邮件任务完成。"