import argparse
import subprocess
import sys

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="依次执行 rq3_inference 并提取最终结果")
    parser.add_argument("--scale", type=str, default="0.15", help="设置 scale 参数 (默认: 0.1)")
    parser.add_argument("--test_dataset", type=str, default="/home/zyc/code/V2XGen/rq3/rq3_test_valid", help="设置 dataset_dir 路径")
    args = parser.parse_args()

    # 定义所有需要遍历的模型配置 (model_dir, fusion_method)
    model_configs = [
        # ("model/early_fusion", "early"),
        # ("model/late_fusion", "late"),
        # ("model/PointPillar_V2VNet", "intermediate"),
        # ("model/PointPillar_V2XViT", "intermediate"),
        ("model/attfuse", "intermediate"),
        # ("model/PointPillar_Fcooper", "intermediate"),
    ]

    # 定义所有需要遍历的方法
    methods = ["ori", "v2x_gen", "coo_test", "random"]
    # methods = ["coo_test"]

    # 依次构造并执行命令
    total_commands = len(model_configs) * len(methods)
    current_cmd = 1

    for model_dir, fusion_method in model_configs:
        for method in methods:
            # 动态拼接命令，将传入的 args.test_dataset 映射给 --dataset_dir
            cmd = (
                f"python opencood/rq_eval/rq3_inference.py "
                f"--scale {args.scale} "
                f"--method {method} "
                f"--dataset_dir {args.test_dataset} "
                f"--model_dir {model_dir} "
                f"--fusion_method {fusion_method}"
            )
            
            print(f"\n[{current_cmd}/{total_commands}] 正在执行: {cmd}")
            
            try:
                # 执行命令并捕获标准输出和标准错误
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                output = result.stdout

                # 提取最后的结果块
                lines = output.strip().split('\n')
                dash_line = "------------------------------------------------------"
                
                # 找到所有出现虚线的行索引
                dash_indices = [idx for idx, line in enumerate(lines) if dash_line in line]
                
                # 根据你提供的格式，最后的结果应该被包含在最后出现的 3 根虚线中
                if len(dash_indices) >= 3:
                    start_idx = dash_indices[-3]
                    end_idx = dash_indices[-1]
                    result_block = "\n".join(lines[start_idx:end_idx+1])
                    print(">>> 执行结果:")
                    print(result_block)
                else:
                    # 如果没有匹配到预期的 3 根虚线，则回退打印最后 10 行以防丢失信息
                    print(">>> 执行结果 (未匹配到完整的虚线格式，输出最后 10 行):")
                    print("\n".join(lines[-10:]))

            except subprocess.CalledProcessError as e:
                print(f"命令执行失败，退出状态码: {e.returncode}")
                print(">>> 错误输出:")
                print(e.stdout)
            
            current_cmd += 1

if __name__ == "__main__":
    main()