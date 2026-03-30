#!/bin/bash
# task_manager.sh

SCRIPT_NAME=$(basename "$0")
TASK_PID_FILE="task.pid"
PROCESS_GROUP_FILE="process_group.id"
# 全局默认任务脚本
DEFAULT_TASK_SCRIPT="sft.sh"

show_usage() {
    echo "任务管理器"
    echo "用法: $SCRIPT_NAME {start|stop|restart|status} [script_name]"
    echo ""
    echo "示例:"
    echo "  ./$SCRIPT_NAME               # 启动默认的 $DEFAULT_TASK_SCRIPT（不带参数默认启动）"
    echo "  ./$SCRIPT_NAME start         # 启动默认的 $DEFAULT_TASK_SCRIPT"
    echo "  ./$SCRIPT_NAME start my.sh   # 启动 my.sh"
    echo "  ./$SCRIPT_NAME stop          # 停止任务"
    echo "  ./$SCRIPT_NAME restart       # 重启任务"
    echo "  ./$SCRIPT_NAME status        # 查看状态"
    exit 1
}

start_task() {
    local TASK_SCRIPT=${1:-"$DEFAULT_TASK_SCRIPT"} 
    
    if [ -f "$TASK_PID_FILE" ]; then
        if kill -0 $(cat "$TASK_PID_FILE") 2>/dev/null; then
            echo "任务已在运行中，PID: $(cat "$TASK_PID_FILE")"
            echo "如果要重启，请先使用: ./$SCRIPT_NAME stop"
            return
        else
            rm -f "$TASK_PID_FILE"
        fi
    fi

    if [ ! -f "$TASK_SCRIPT" ]; then
        echo "错误: 找不到脚本文件 '$TASK_SCRIPT'"
        echo "请确保脚本文件存在，或指定正确的脚本名称"
        echo "用法: ./$SCRIPT_NAME [start] [script_name]"
        return 1
    fi

    # 生成日志文件名
    LOG_FILE="logs/log_$(date +%Y%m%d_%H%M%S).log"
    echo "日志文件: $LOG_FILE"

    # 创建新的进程组并启动任务
    echo "启动任务: $TASK_SCRIPT"
    setsid bash "$TASK_SCRIPT" > "$LOG_FILE" 2>&1 &
    MAIN_PID=$!

    # 记录主进程PID和进程组ID
    echo "$MAIN_PID" > "$TASK_PID_FILE"
    PGID=$(ps -o pgid= -p "$MAIN_PID" | tr -d ' ')
    echo "$PGID" > "$PROCESS_GROUP_FILE"
    
    echo "========================================="
    echo "任务 '$TASK_SCRIPT' 已启动"
    echo "主进程 PID: $MAIN_PID"
    echo "进程组 ID: $PGID"
    echo "日志文件: $LOG_FILE"
    echo "========================================="
    
    # 等待一下，检查进程是否成功启动
    sleep 1
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "警告: 进程可能启动失败，请检查日志文件: $LOG_FILE"
        rm -f "$TASK_PID_FILE" "$PROCESS_GROUP_FILE"
        return 1
    fi
    
    echo -e "\n管理命令:"
    echo "  ./$SCRIPT_NAME stop    # 停止任务"
    echo "  ./$SCRIPT_NAME status  # 查看状态"
    echo "  ./$SCRIPT_NAME restart # 重启任务"
    
    return 0
}

stop_task() {
    if [ ! -f "$PROCESS_GROUP_FILE" ] || [ ! -f "$TASK_PID_FILE" ]; then
        echo "没有找到正在运行的任务"
        return
    fi

    PGID=$(cat "$PROCESS_GROUP_FILE")
    MAIN_PID=$(cat "$TASK_PID_FILE")

    echo "正在停止进程组 $PGID..."
    
    # 向整个进程组发送终止信号
    if kill -TERM -"$PGID" 2>/dev/null; then
        echo "已发送终止信号到进程组 $PGID"
        sleep 2
        
        # 检查是否还有存活的进程
        if ps -p "$MAIN_PID" >/dev/null 2>&1; then
            echo "进程仍在运行，发送KILL信号..."
            kill -KILL -"$PGID" 2>/dev/null
        fi
    else
        echo "进程组 $PGID 已停止"
    fi

    # 清理PID文件
    rm -f "$TASK_PID_FILE" "$PROCESS_GROUP_FILE"
    echo "任务已完全停止"
}

show_status() {
    if [ ! -f "$TASK_PID_FILE" ]; then
        echo "任务未运行"
        return
    fi

    MAIN_PID=$(cat "$TASK_PID_FILE")
    if kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "任务正在运行"
        echo "主进程 PID: $MAIN_PID"
        echo "进程组 ID: $(cat "$PROCESS_GROUP_FILE" 2>/dev/null || echo '未知')"
        
        # 显示相关进程
        PGID=$(cat "$PROCESS_GROUP_FILE" 2>/dev/null)
        if [ -n "$PGID" ]; then
            echo -e "\n进程组中的进程:"
            ps -o pid,ppid,pgid,cmd -g "$PGID" 2>/dev/null || echo "无法获取进程列表"
        fi
    else
        echo "任务已停止（PID文件存在但进程不存在）"
        rm -f "$TASK_PID_FILE" "$PROCESS_GROUP_FILE"
    fi
}

# 主逻辑
if [ $# -eq 0 ]; then
    # 没有参数时，默认启动DEFAULT_TASK_SCRIPT
    start_task "$DEFAULT_TASK_SCRIPT"
    exit $?
fi

case "$1" in
    start)
        TASK_SCRIPT=${2:-"$DEFAULT_TASK_SCRIPT"}
        start_task "$TASK_SCRIPT"
        ;;
    stop)
        stop_task
        ;;
    restart)
        stop_task
        sleep 1
        TASK_SCRIPT=${2:-"$DEFAULT_TASK_SCRIPT"}
        start_task "$TASK_SCRIPT"
        ;;
    status)
        show_status
        ;;
    *)
        echo "未知命令: $1"
        show_usage
        ;;
esac