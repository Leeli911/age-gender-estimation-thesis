# run_all.py
import os
import subprocess
import datetime

# 创建 logs 汇总目录
summary_dir = "summary_logs"
os.makedirs(summary_dir, exist_ok=True)

# 记录每个脚本及对应日志文件
experiments = [
    ("train_age76_gender.py", "train_age76_gender/train.log"),
    ("train_age76_nongender.py", "logs_age76_nogender/train.log"),
    ("train_group5_gender.py", "logs_group5_gender/train.log"),
    ("train_group5_nongender.py", "logs_group5_nogender/train.log")
]

summary_file = os.path.join(summary_dir, f"summary_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

with open(summary_file, "w") as summary:
    for script, log_path in experiments:
        print(f"\n[INFO] Running {script}...")
        summary.write(f"===== {script} =====\n")

        try:
            # 运行脚本
            subprocess.run(["python", script], check=True)

            # 提取日志中最新的 Test MAE
            if os.path.exists(log_path):
                with open(log_path, "r") as log_file:
                    lines = log_file.readlines()
                    test_mae_lines = [l for l in lines if "Test MAE" in l]
                    if test_mae_lines:
                        summary.write(test_mae_lines[-1])
                        print(test_mae_lines[-1].strip())
                    else:
                        summary.write("Test MAE not found.\n")
                        print("[WARNING] Test MAE not found in log.")
            else:
                summary.write("Log file not found.\n")
                print("[WARNING] Log file not found.")

        except subprocess.CalledProcessError:
            summary.write(f"{script} failed to run.\n")
            print(f"[ERROR] {script} execution failed.")

print(f"\n All experiments finished. Summary saved to: {summary_file}")
