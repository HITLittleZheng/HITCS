import argparse
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import createData
from selectk.randomSelect import randomSelect


def get_args():
    parser = argparse.ArgumentParser(description='select k algorithm')

    parser.add_argument(
        '--sizes',
        type=str,
        default='50000',
        help='数据大小，例如 50000'
    )

    parser.add_argument(
        '--sample_method',
        type=str,
        default='normal',
        help='采样方法 uniform/normal'
    )

    parser.add_argument(
        '--sample_rates',
        type=str,
        default='0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0',
        help='采样参数列表'
    )

    parser.add_argument(
        '--repeat',
        type=int,
        default=1000,
        help='每个采样参数重复次数'
    )

    return parser.parse_args()


def run_algorithm_once(dataset, k, sample_rate):
    """
    只运行一次随机选择算法。
    如果这一次失败，randomSelect 返回 -1。
    """
    start_time = time.time()

    k_value, retry_times = randomSelect(
        dataset,
        k,
        sample_rate=sample_rate,
        test_once=True
    )

    end_time = time.time()

    run_time_ms = (end_time - start_time) * 1000

    return run_time_ms, k_value, retry_times


if __name__ == '__main__':
    args = get_args()

    sizes = list(map(int, args.sizes.split(',')))
    sample_rates = list(map(float, args.sample_rates.split(',')))
    repeat = args.repeat

    datasets = createData(sizes, sample_method=args.sample_method)

    all_success_probs = []
    all_times = []

    for rate in tqdm(sample_rates, desc="Testing sample rates"):
        success_probs = []
        times = []

        for dataset in tqdm(datasets, desc=f"Testing datasets for rate {rate}", leave=False):
            n = len(dataset)

            # 每个数据集固定一个 k
            k = random.randint(0, n - 1)

            # 正确答案只排序一次，不能放进 1000 次循环
            correct_value = sorted(dataset)[k]

            run_time_sum = 0
            success_count = 0

            for _ in tqdm(range(repeat), desc=f"Running {repeat} times, n={n}", leave=False):
                run_time, k_value, _ = run_algorithm_once(dataset, k, sample_rate=rate)

                run_time_sum += run_time

                if k_value == correct_value:
                    success_count += 1

            avg_success_prob = success_count / repeat
            avg_time = run_time_sum / repeat

            success_probs.append(avg_success_prob)
            times.append(avg_time)

        avg_success_prob = sum(success_probs) / len(success_probs)
        avg_time = sum(times) / len(times)

        all_success_probs.append(avg_success_prob)
        all_times.append(avg_time)

        print(f"\nsample_rate = {rate}")
        print(f"一次运行成功概率 = {avg_success_prob:.4f}")
        print(f"平均单次运行时间 = {avg_time:.4f} ms")

    # 图1：成功概率对比
    plt.figure(figsize=(12, 6))
    plt.plot(sample_rates, all_success_probs, '-o', label='Success Probability')
    plt.xlabel("Sample Parameter a")
    plt.ylabel("Success Probability")
    plt.title("Success Probability vs Sample Parameter")
    plt.grid(True)
    plt.legend()
    plt.xticks(sample_rates)
    plt.savefig("Success_Probability_Curve_for_random.png", dpi=300)
    plt.close()

    # 图2：运行时间对比
    plt.figure(figsize=(12, 6))
    plt.plot(sample_rates, all_times, '-o', label='Running Time (ms)')
    plt.xlabel("Sample Parameter a")
    plt.ylabel("Running Time (ms)")
    plt.title("Running Time vs Sample Parameter")
    plt.grid(True)
    plt.legend()
    plt.xticks(sample_rates)
    plt.savefig("Running_Time_Curve_for_random.png", dpi=300)
    plt.close()

    print("\n实验完成！结果已保存为：")
    print("  - Success_Probability_Curve_for_random.png")
    print("  - Running_Time_Curve_for_random.png")