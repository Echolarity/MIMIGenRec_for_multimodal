#!/bin/bash
# nohup_run.sh

# 生成日志文件名
LOG_FILE="log_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"

# 启动任务
nohup bash evaluate.sh > "$LOG_FILE" 2>&1 &
TASK_PID=$!

# 记录PID
echo "$TASK_PID" > task.pid
echo "任务已启动，PID: $TASK_PID"

