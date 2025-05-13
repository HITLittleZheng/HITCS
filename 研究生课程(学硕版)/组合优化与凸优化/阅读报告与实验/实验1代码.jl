using JuMP
using LinearAlgebra
using Plots
using Random

# 设置随机种子以确保实验可重复
Random.seed!(1234)

# 定义Rosenbrock函数及其一阶、二阶导数
# Rosenbrock函数是优化领域著名的测试函数，形状呈弯曲的“香蕉谷”，具有一个窄而弯曲的最优路径，考验优化算法的收敛性和稳健性。

# 目标函数
function rosenbrock(x)
    return sum(100 * (x[2i] - x[2i-1]^2)^2 + (1 - x[2i-1])^2 for i in 1:(length(x)÷2))
end

# 梯度
function rosenbrock_gradient(x)
    n = length(x)
    g = zeros(n)
    for i in 1:2:n-1
        g[i] = -400 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
        g[i+1] = 200 * (x[i+1] - x[i]^2)
    end
    return g
end

# Hessian矩阵
function rosenbrock_hessian(x)
    n = length(x)
    H = zeros(n, n)
    for i in 1:2:n-1
        H[i,i] = 1200 * x[i]^2 - 400 * x[i+1] + 2
        H[i,i+1] = -400 * x[i]
        H[i+1,i] = -400 * x[i]
        H[i+1,i+1] = 200
    end
    return H
end

# 1. Nelder-Mead单纯形法 (无梯度方法)
# 通过维护一个简单形（n+1个点）不断地反射、扩展、收缩来逼近最优解，适合无梯度优化。
function nelder_mead(f, x0; max_iter=1000, tol=1e-6)
    n = length(x0)
    simplex = [x0 + 0.1 * randn(n) for _ in 1:(n+1)]
    values = [f(s) for s in simplex]
    history = [copy(x0)]
    
    for iter in 1:max_iter
        idx = sortperm(values)
        simplex = simplex[idx]
        values = values[idx]
        
        x_best, x_worst = simplex[1], simplex[end]
        f_best, f_worst = values[1], values[end]
        
        # 计算除最差点外的质心
        x_centroid = sum(simplex[1:end-1]) / n
        
        # 反射步骤
        x_reflect = x_centroid + (x_centroid - x_worst)
        f_reflect = f(x_reflect)
        
        if f_reflect < f_best
            # 扩展步骤
            x_expand = x_centroid + 2 * (x_reflect - x_centroid)
            f_expand = f(x_expand)
            if f_expand < f_reflect
                simplex[end] = x_expand
                values[end] = f_expand
            else
                simplex[end] = x_reflect
                values[end] = f_reflect
            end
        elseif f_reflect < values[end-1]
            # 接受反射点
            simplex[end] = x_reflect
            values[end] = f_reflect
        else
            # 收缩步骤
            x_contract = x_centroid + 0.5 * (x_worst - x_centroid)
            f_contract = f(x_contract)
            if f_contract < f_worst
                simplex[end] = x_contract
                values[end] = f_contract
            else
                # 全局缩小
                for i in 2:n+1
                    simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                end
            end
        end
        
        push!(history, copy(simplex[1]))
        if maximum(values) - minimum(values) < tol
            break
        end
    end
    return simplex[1], history
end

# 2. 动量梯度下降 (Momentum GD)
# 在标准梯度下降基础上加动量项，加速收敛并抑制震荡。
function gradient_descent(f, grad, x0; max_iter=5000, tol=1e-6, β=0.9)
    x = copy(x0)
    v = zeros(length(x0))  # 动量向量
    history = [copy(x)]
    
    for iter in 1:max_iter
        g = grad(x)
        v = β * v - (1 - β) * g  # 更新动量
        d = v
        
        # 使用Armijo线搜索确定步长
        α = 0.1
        c = 0.5
        ρ = 0.5
        fx = f(x)
        
        while f(x + α * d) > fx + c * α * dot(g, d)
            α *= ρ
            if α < 1e-10
                break
            end
        end
        
        x += α * d
        push!(history, copy(x))
        if norm(g) < tol
            break
        end
    end
    return x, history
end

# 3. 共轭梯度法 (CG)
# 适用于二次优化问题，通过构建共轭方向序列快速收敛，比标准梯度下降快。
function conjugate_gradient(f, grad, x0; max_iter=1000, tol=1e-6)
    x = copy(x0)
    g = grad(x)
    d = -g
    history = [copy(x)]
    
    for iter in 1:max_iter
        α = 0.1
        c = 0.5
        ρ = 0.5
        fx = f(x)
        
        while f(x + α * d) > fx + c * α * dot(g, d)
            α *= ρ
            if α < 1e-10
                break
            end
        end
        
        x += α * d
        g_new = grad(x)
        β = dot(g_new, g_new) / dot(g, g)
        d = -g_new + β * d
        g = g_new
        push!(history, copy(x))
        if norm(g) < tol
            break
        end
    end
    return x, history
end

# 4. 牛顿法 (Newton Method)
# 使用Hessian矩阵的二阶信息，局部收敛速度是平方的（非常快），但对初值敏感。
function newton_method(f, grad, hess, x0; max_iter=100, tol=1e-6)
    x = copy(x0)
    history = [copy(x)]
    
    for iter in 1:max_iter
        g = grad(x)
        H = hess(x)
        if cond(H) > 1e10
            break  # 避免病态Hessian
        end
        d = -H \ g
        x += d
        push!(history, copy(x))
        if norm(g) < tol
            break
        end
    end
    return x, history
end

# 5. 增广拉格朗日法 (ADMM)
# 将问题分解成子问题交替求解，适合复杂约束和分布式优化。
function admm(f, grad, x0; max_iter=1000, tol=1e-6, ρ=10.0)
    x = copy(x0)
    z = copy(x0)
    u = zeros(length(x0))
    history = [copy(x)]
    
    for iter in 1:max_iter
        # x-update
        x_new = copy(x)
        for _ in 1:5  # 少量牛顿迭代
            g = grad(x_new) + ρ * (x_new - z + u)
            H = rosenbrock_hessian(x_new) + ρ * I
            if cond(H) > 1e10
                break
            end
            d = -H \ g
            x_new += 0.5 * d
            if norm(d) < 1e-8
                break
            end
        end
        x = x_new
        
        # z-update (简单近似)
        z_new = copy(z)
        z_new[1] = (ρ * (x[1] + u[1]) + 2) / (ρ + 2)
        z_new[2] = x[2] + u[2]
        z = z_new
        
        # u-update
        u += x - z
        
        push!(history, copy(x))
        if norm(x - z) < tol
            break
        end
    end
    return x, history
end

# 6. Krylov子空间方法（以共轭梯度形式）
# 在子空间内寻找下降方向，适合大型稀疏系统。
function krylov_cg(f, grad, x0; max_iter=100, tol=1e-6)
    return conjugate_gradient(f, grad, x0; max_iter=max_iter, tol=tol)
end

# 主程序
function main()
    x0 = [2.0, 2.0]  # 初始点
    methods = [
        ("Nelder-Mead", nelder_mead, rosenbrock),
        ("Gradient Descent", gradient_descent, rosenbrock, rosenbrock_gradient),
        ("Conjugate Gradient", conjugate_gradient, rosenbrock, rosenbrock_gradient),
        ("Newton", newton_method, rosenbrock, rosenbrock_gradient, rosenbrock_hessian),
        ("ADMM", admm, rosenbrock, rosenbrock_gradient),
        ("Krylov CG", krylov_cg, rosenbrock, rosenbrock_gradient)
    ]
    
    results = []
    for (name, method, args...) in methods
        x_opt, history = method(args..., x0)
        push!(results, (name, history))
        println("$name solution: $x_opt")
    end

    # 使用subplot绘制各个方法的收敛曲线
    p = plot(layout=(3,2), size=(1000,800), title="Optimization Methods on Rosenbrock")
    for (i, (name, history)) in enumerate(results)
        f_vals = [rosenbrock(x) for x in history]
        plot!(p[i], 1:length(f_vals), f_vals, label=name, lw=2, xlabel="Iteration", ylabel="Function Value", title=name)
    end
    savefig(p, "subplots_convergence.png")
    println("Plot saved as subplots_convergence.png")
end

# 运行
main()
