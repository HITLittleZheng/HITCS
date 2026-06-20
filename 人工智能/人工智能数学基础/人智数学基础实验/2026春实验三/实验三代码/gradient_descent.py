import torch
import matplotlib.pyplot as plt
import numpy as np
import time

#  定义测试函数列表 
def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rastrigin(x, y):
    A = 10
    return A*2 + (x**2 - A*torch.cos(2*np.pi*x)) + (y**2 - A*torch.cos(2*np.pi*y))

def booth(x, y):
    """Booth 函数，全局最小 f(1,3)=0"""
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

# 函数列表：(函数, 名称, 显示边界, 各优化器的学习率)
functions = [
    (sphere, "Sphere", (-2, 2), {'sgd': 0.01, 'momentum': 0.01, 'adam': 0.01}),
    (rosenbrock, "Rosenbrock", (-2, 2), {'sgd': 0.001, 'momentum': 0.001, 'adam': 0.01}),
    (rastrigin, "Rastrigin", (-2, 2), {'sgd': 0.01, 'momentum': 0.01, 'adam': 0.01}),
    (booth, "Booth", (-5, 5), {'sgd': 0.01, 'momentum': 0.01, 'adam': 0.01})
]

# 固定初始点（用于观察收敛区域）
start_points = [(0.5, 1.0)]

#  优化器实现 
def sgd_step(params, grads, lr):
    for p, g in zip(params, grads):
        p.data -= lr * g

def momentum_step(params, grads, velocities, lr, momentum=0.9):
    for i, (p, g) in enumerate(zip(params, grads)):
        v = velocities[i]
        v.data = momentum * v + lr * g
        p.data -= v

def adam_step(params, grads, states, step, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]['m'], states[i]['v']
        m.data = beta1 * m + (1 - beta1) * g
        v.data = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)

def gradient_clipping(grads, max_norm=1.0):
    total_norm = torch.sqrt(sum(torch.norm(g)**2 for g in grads))
    if total_norm > max_norm:
        factor = max_norm / total_norm
        for g in grads:
            g.data *= factor

#  运行优化并记录路径、迭代次数、时间 
def optimize(func, x0, y0, optimizer='sgd', lr=0.01, max_iter=1000, clip_norm=1.0):
    start_time = time.time()
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y0, dtype=torch.float32, requires_grad=True)
    params = [x, y]
    path = [(x.item(), y.item())]
    
    if optimizer == 'momentum':
        velocities = [torch.zeros_like(p) for p in params]
    elif optimizer == 'adam':
        states = [{'m': torch.zeros_like(p), 'v': torch.zeros_like(p)} for p in params]
        step = 0
    
    n_iter = 0
    for i in range(max_iter):
        loss = func(x, y)
        grads = torch.autograd.grad(loss, params, create_graph=False)
        if clip_norm > 0:
            gradient_clipping(grads, clip_norm)
        
        if optimizer == 'sgd':
            sgd_step(params, grads, lr)
        elif optimizer == 'momentum':
            momentum_step(params, grads, velocities, lr)
        elif optimizer == 'adam':
            step += 1
            adam_step(params, grads, states, step, lr)
        
        xv, yv = x.item(), y.item()
        if np.isnan(xv) or np.isnan(yv) or np.isinf(xv) or np.isinf(yv):
            elapsed = time.time() - start_time
            return None, n_iter, elapsed  # 失败返回 None 路径
        path.append((xv, yv))
        n_iter += 1
        
        if all(torch.norm(g).item() < 1e-5 for g in grads):
            break
    
    elapsed = time.time() - start_time
    return path, n_iter, elapsed

#  计算等高线数据（复用） 
def get_contour_data(func, bounds, grid_size=400):
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = []
    for xi, yi in zip(X, Y):
        row = []
        for xx, yy in zip(xi, yi):
            val = func(torch.tensor(xx), torch.tensor(yy)).detach().numpy()
            row.append(val)
        Z.append(row)
    return X, Y, np.array(Z)

#  可视化：每个函数一张大图，并排三个子图，标题包含平均迭代次数和时间 
def plot_optimizers_comparison(func, func_name, bounds, optimizers_config, start_points, lr_dict):
    X, Y, Z = get_contour_data(func, bounds)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{func_name} function - Optimization paths comparison', fontsize=14)
    
    # 存储每个优化器的统计信息，用于子图标题
    opt_stats = {}
    
    for idx, (opt_name, opt_key) in enumerate(optimizers_config.items()):
        ax = axes[idx]
        contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        lr = lr_dict.get(opt_key, 0.01)
        iter_list = []
        time_list = []
        
        for (x0, y0) in start_points:
            path, n_iter, elapsed = optimize(func, x0, y0, optimizer=opt_key, lr=lr, max_iter=500, clip_norm=1.0)
            if path is None:
                continue
            iter_list.append(n_iter)
            time_list.append(elapsed)
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'o-', linewidth=1.2, markersize=2, alpha=0.8,
                    label=f'from ({x0},{y0})')
            ax.plot(path[0,0], path[0,1], 'go', markersize=6, alpha=0.8)
            ax.plot(path[-1,0], path[-1,1], 'ro', markersize=6, alpha=0.8)
        
        avg_iter = np.mean(iter_list) if iter_list else 0
        avg_time = np.mean(time_list) if time_list else 0
        opt_stats[opt_name] = (avg_iter, avg_time)
        ax.set_title(f'{opt_name} (lr={lr})\nAvg iter={avg_iter:.1f}, Avg time={avg_time:.4f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.show()
    return opt_stats

#  主实验 
if __name__ == '__main__':
    optimizers_config = {'SGD': 'sgd', 'Momentum': 'momentum', 'Adam': 'adam'}
    
    for func, func_name, bounds, lr_dict in functions:
        print(f"\n========== 测试函数: {func_name} ==========")
        # 绘制并排对比图，并获取平均统计信息
        stats = plot_optimizers_comparison(func, func_name, bounds, optimizers_config, start_points, lr_dict)
        
        # 打印收敛区域分析（包含详细迭代次数和时间）
        print(f"收敛区域分析（最终点坐标、迭代次数、时间） - {func_name}:")
        for opt_name, opt_key in optimizers_config.items():
            lr = lr_dict.get(opt_key, 0.01)
            print(f"  {opt_name}:")
            for (x0, y0) in start_points:
                path, n_iter, elapsed = optimize(func, x0, y0, optimizer=opt_key, lr=lr, max_iter=1000, clip_norm=1.0)
                if path is None:
                    print(f"    起点({x0},{y0}) -> 优化失败")
                else:
                    final = path[-1]
                    print(f"    起点({x0},{y0}) -> 终点({final[0]:.4f},{final[1]:.4f}) | 迭代次数:{n_iter} | 时间:{elapsed:.4f}s")