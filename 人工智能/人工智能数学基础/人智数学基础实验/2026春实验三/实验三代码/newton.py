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

# 函数列表：(函数, 名称, 显示边界, 各优化器的学习率/参数)
functions = [
    (sphere, "Sphere", (-2, 2), {'newton': 1.0, 'cubic_newton': 1.0}),
    (rosenbrock, "Rosenbrock", (-2, 2), {'newton': 1.0, 'cubic_newton': 1.0}),
    (rastrigin, "Rastrigin", (-2, 2), {'newton': 0.5, 'cubic_newton': 0.5}) , # 正则化强度可能需要调整
    (booth, "Booth", (-5, 5), {'sgd': 0.01, 'momentum': 0.01, 'adam': 0.01})
]

start_points = [ (0.2, 1.3)]

#  牛顿法（带正则化，保证 Hessian 正定）
def newton_step(x, y, func, lr=1.0, reg=1e-6):
    """
    计算牛顿步： x_new = x - lr * (H + reg*I)^{-1} * g
    返回更新后的参数
    """
    params = torch.tensor([x, y], requires_grad=True)
    loss = func(params[0], params[1])
    # 梯度
    grads = torch.autograd.grad(loss, params, create_graph=True)[0]
    # Hessian 矩阵 (2x2)
    hess = torch.zeros(2, 2, dtype=params.dtype)
    for i in range(2):
        grad_i = grads[i]
        hess_i = torch.autograd.grad(grad_i, params, retain_graph=True)[0]
        hess[i] = hess_i
    # 正则化
    hess_reg = hess + reg * torch.eye(2)
    try:
        step = torch.linalg.solve(hess_reg, grads)  # 解 H * step = g
    except:
        step = torch.linalg.lstsq(hess_reg, grads).solution
    x_new = x - lr * step[0].item()
    y_new = y - lr * step[1].item()
    return x_new, y_new

# 三次正则化牛顿法（求解 min_p f+g^T p + 1/2 p^T H p + sigma/3 ||p||^3）
def cubic_newton_step(x, y, func, sigma=1.0, **kwargs):
    """
    使用 pytorch-minimize 库求解三次正则化子问题。
    该方法利用信赖域精确算法 (trust-exact)，能有效处理非凸问题。
    """
    import torch
    from torchmin import minimize

    def objective(params):
        """目标函数，输入为张量，返回标量损失。"""
        return func(params[0], params[1])

    # 初始点
    x0 = torch.tensor([x, y], requires_grad=True)

    # 调用优化器，这里使用信赖域精确算法，它等价于一种自适应正则化的牛顿方法
    result = minimize(
        objective,
        x0,
        method='trust-exact',   # 信赖域精确算法，鲁棒性强
        max_iter=100,           # 最大迭代次数
        tol=1e-6,               # 收敛容忍度
        disp=0                  # 不打印详细信息
    )

    # 提取优化结果
    x_new = result.x[0].item()
    y_new = result.x[1].item()
    return x_new, y_new

#  通用优化函数（记录路径、迭代次数、时间）
def optimize_newton(func, x0, y0, method='newton', lr=1.0, sigma=1.0, max_iter=100, tol=1e-6):
    """
    method: 'newton' 或 'cubic_newton'
    """
    start_time = time.time()
    x, y = x0, y0
    path = [(x, y)]
    n_iter = 0
    for i in range(max_iter):
        # 计算当前点的梯度范数（用于提前终止）
        params = torch.tensor([x, y], requires_grad=True)
        loss = func(params[0], params[1])
        grad = torch.autograd.grad(loss, params)[0]
        grad_norm = grad.norm().item()
        if grad_norm < tol:
            break
        
        if method == 'newton':
            x, y = newton_step(x, y, func, lr=lr, reg=1e-6)
        else:  # cubic_newton
            x, y = cubic_newton_step(x, y, func, sigma=sigma, max_iter_sub=50)
        
        # 检查数值稳定性
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            elapsed = time.time() - start_time
            return None, n_iter, elapsed
        path.append((x, y))
        n_iter += 1
    elapsed = time.time() - start_time
    return path, n_iter, elapsed

#  计算等高线数据 
def get_contour_data(func, bounds, grid_size=200):
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i,j] = func(torch.tensor(X[i,j]), torch.tensor(Y[i,j])).item()
    return X, Y, Z

#  可视化：并排对比牛顿法和三次正则化牛顿法 
def plot_newton_comparison(func, func_name, bounds, start_points, method_params):
    """
    method_params: dict，例如 {'Newton': {'method':'newton','lr':1.0}, 
                               'Cubic Newton': {'method':'cubic_newton','sigma':1.0}}
    """
    X, Y, Z = get_contour_data(func, bounds, grid_size=200)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{func_name} function - Newton methods comparison', fontsize=14)
    
    stats = {}
    for idx, (method_name, params) in enumerate(method_params.items()):
        ax = axes[idx]
        contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        iter_list = []
        time_list = []
        for (x0, y0) in start_points:
            if params['method'] == 'newton':
                path, n_iter, elapsed = optimize_newton(func, x0, y0, method='newton', 
                                                        lr=params.get('lr',1.0), max_iter=100)
            else:
                path, n_iter, elapsed = optimize_newton(func, x0, y0, method='cubic_newton',
                                                        sigma=params.get('sigma',1.0), max_iter=100)
            iter_list.append(n_iter)
            time_list.append(elapsed)
            path = np.array(path)
            ax.plot(path[:,0], path[:,1], 'o-', linewidth=1.2, markersize=2, alpha=0.8,
                    label=f'from ({x0},{y0})')
            ax.plot(path[0,0], path[0,1], 'go', markersize=6, alpha=0.8)
            ax.plot(path[-1,0], path[-1,1], 'ro', markersize=6, alpha=0.8)
        
        avg_iter = np.mean(iter_list) if iter_list else 0
        avg_time = np.mean(time_list) if time_list else 0
        stats[method_name] = (avg_iter, avg_time)
        ax.set_title(f'{method_name}\nAvg iter={avg_iter:.1f}, Avg time={avg_time:.4f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.show()
    return stats

# 主实验 
if __name__ == '__main__':
    # 定义两种牛顿方法的参数
    newton_methods = {
        'Newton': {'method': 'newton', 'lr': 1.0},
        'Cubic Newton': {'method': 'cubic_newton', 'sigma': 1.0}
    }
    
    for func, func_name, bounds, param_dict in functions:
        # 从 param_dict 中提取对应方法的参数（如果有定制）
        # 这里简单覆盖：如果函数字典中有 'newton' 键，则作为 lr；有 'cubic_newton' 作为 sigma
        newton_methods['Newton']['lr'] = param_dict.get('newton', 1.0)
        newton_methods['Cubic Newton']['sigma'] = param_dict.get('cubic_newton', 1.0)
        
        print(f"\n========== 测试函数: {func_name} ==========")
        stats = plot_newton_comparison(func, func_name, bounds, start_points, newton_methods)
        
        # 打印详细收敛信息
        print(f"收敛区域分析（最终点坐标、迭代次数、时间） - {func_name}:")
        for method_name, params in newton_methods.items():
            print(f"  {method_name}:")
            for (x0, y0) in start_points:
                if params['method'] == 'newton':
                    path, n_iter, elapsed = optimize_newton(func, x0, y0, method='newton',
                                                            lr=params['lr'], max_iter=100)
                else:
                    path, n_iter, elapsed = optimize_newton(func, x0, y0, method='cubic_newton',
                                                            sigma=params['sigma'], max_iter=100)
                if path is None:
                    print(f"    起点({x0},{y0}) -> 优化失败")
                else:
                    final = path[-1]
                    print(f"    起点({x0},{y0}) -> 终点({final[0]:.4f},{final[1]:.4f}) | 迭代次数:{n_iter} | 时间:{elapsed:.4f}s")
        print(f"\n  平均性能汇总:")
        for method_name, (avg_iter, avg_time) in stats.items():
            print(f"    {method_name}: 平均迭代次数={avg_iter:.1f}, 平均时间={avg_time:.4f}s")