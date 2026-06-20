"""精简版：训练 SGD / Momentum / Adam 并生成 2×3 指标对比图"""
import os, sys, json, glob, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam
from deepobs import pytorch as pt

# ---------- 配置 ----------
OUTPUT_DIR = "./results"
NUM_EPOCHS = 20
parser = argparse.ArgumentParser()
parser.add_argument("testproblem", default="fmnist_2c3d", nargs="?")
parser.add_argument("-N", "--num_epochs", type=int, default=NUM_EPOCHS)
parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR)
args = parser.parse_args()
print(f"Problem: {args.testproblem}, Epochs: {args.num_epochs}, Output: {args.output_dir}")

# ---------- 优化器配置 ----------
configs = [
    ("SGD", SGD,
     {"lr": 0.1, "momentum": 0.0},
     {"lr": {"type": float, "default": 0.1},
      "momentum": {"type": float, "default": 0.0}}),

    ("Momentum", SGD,
     {"lr": 0.01, "momentum": 0.99},
     {"lr": {"type": float, "default": 0.01},
      "momentum": {"type": float, "default": 0.99},
      "nesterov": {"type": bool, "default": False}}),

    ("Adam", Adam,
     {"lr": 0.001},
     {"lr": {"type": float, "default": 0.001},
      "betas": {"type": tuple, "default": (0.9, 0.999)},
      "eps": {"type": float, "default": 1e-8}}),
]

# ---------- 训练 ----------
for name, opt_cls, hparams, param_decl in configs:
    print(f"\n>>> Training {name} ...")
    runner = pt.runners.StandardRunner(opt_cls, param_decl)
    sys.argv = [sys.argv[0], args.testproblem,
                "-N", str(args.num_epochs),
                f"--output_dir={args.output_dir}"] + \
               [f"--{k}={v}" for k, v in hparams.items()]
    runner.run()
    print(f"  {name} finished")

# ---------- 加载结果（基于 JSON 内容匹配，避免目录格式依赖）----------
def params_match(json_params, expected):
    """宽松比较超参数，处理数值类型和元组"""
    for k, v in expected.items():
        if k not in json_params:
            return False
        jv = json_params[k]
        if isinstance(v, (int, float)) and isinstance(jv, (int, float)):
            if float(v) != float(jv):
                return False
        elif isinstance(v, tuple):
            if list(jv) != list(v):
                return False
        elif jv != v:
            return False
    return True

def load_results():
    prob_dir = os.path.join(args.output_dir, args.testproblem)
    if not os.path.isdir(prob_dir):
        return {}
    all_json = glob.glob(os.path.join(prob_dir, "**/*.json"), recursive=True)
    results = {}

    for name, opt_cls, hparams, _ in configs:
        best_fp = None
        best_mtime = -1
        best_data = None

        for fp in all_json:
            with open(fp) as f:
                data = json.load(f)
            # 检查优化器类型和超参数是否匹配
            if (data.get("optimizer_name") == opt_cls.__name__ and
                params_match(data.get("optimizer_hyperparams", {}), hparams)):
                mtime = os.path.getmtime(fp)
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_fp = fp
                    best_data = data

        if best_data is not None:
            results[name] = best_data
            print(f"  Loaded {name:10s} (latest) <- {os.path.basename(best_fp)}")
        else:
            print(f"  Missing {name}: {opt_cls.__name__} with {hparams}")

    return results

results = load_results()
if not results:
    sys.exit("No results loaded.")

# ---------- 绘图 ----------
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
metrics = [
    ("train_losses", "Train Loss"), ("valid_losses", "Valid Loss"), ("test_losses", "Test Loss"),
    ("train_accuracies", "Train Acc (%)"), ("valid_accuracies", "Valid Acc (%)"), ("test_accuracies", "Test Acc (%)")
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for idx, (key, title) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    for i, (name, data) in enumerate(results.items()):
        vals = data.get(key, [])
        if vals:
            if "acc" in key:
                vals = [v * 100 for v in vals]
            ax.plot(range(len(vals)), vals, label=name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if "loss" in key else "Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
out_img = os.path.join(args.output_dir, "comparison_all_metrics.png")
plt.savefig(out_img, dpi=150)
print(f"\n✅ Saved: {out_img}")
