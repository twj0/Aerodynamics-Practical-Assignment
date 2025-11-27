### Aerodynamics-Practical-Assignment 仓库使用指南

~~当时对待作业真的好认真~~

这个仓库提供了空气动力学中平板边界层(Blasius)方程的数值求解程序。通过Python代码实现Blasius方程的求解，并将结果可视化和导出为Excel文件。

### 仓库内容概述

- `ode.py`: 核心Python代码，实现Blasius方程求解
- `excel/`: 存放生成的Excel结果文件
- `fig/`: 存放生成的可视化图像
- `实践作业.jpg`: 作业图片
- `practice_work1.code-workspace`: VS Code工作区配置文件

### 环境准备

在使用本仓库代码前，请确保安装以下依赖:

- Python 3.6+
- NumPy: 用于数值计算
- Matplotlib: 用于结果可视化
- Pandas: 用于数据处理和Excel导出

可以通过以下命令安装所需依赖:

```bash
pip install numpy matplotlib pandas
```

### 使用方法

1. **克隆仓库到本地**

```bash
git clone https://github.com/your-username/Aerodynamics-Practical-Assignment.git
cd Aerodynamics-Practical-Assignment
```

2. **运行代码**

直接执行Python脚本:

```bash
python ode.py
```

或者在IDE中打开并运行`ode.py`文件。

3. **查看结果**

- 程序会在控制台输出计算过程和最终结果
- Excel文件会保存在`excel/`目录下
- 图像会保存在`fig/`目录下

### ode.py代码详解

`ode.py`是整个项目的核心文件，主要实现了以下功能:

#### 1. Blasius方程求解原理

Blasius方程是描述层流边界层的经典非线性常微分方程:

```
f''' + 0.5 * f * f'' = 0
```

边界条件:
- f(0) = 0
- f'(0) = 0
- f'(∞) = 1

#### 2. 主要函数说明

- **`blasius_ode(eta, y)`**: 将三阶Blasius方程转化为一阶常微分方程组
- **`runge_kutta4(ode_func, y0, eta_values)`**: 实现四阶龙格-库塔法求解常微分方程
- **`solve_blasius(eta_max, d_eta, f_double_prime_guess)`**: 使用打靶法求解Blasius方程
- **`calculate_friction(f_double_prime_0, rho, U_inf, L)`**: 计算壁面摩擦系数和阻力
- **`main()`**: 主函数，协调所有计算和结果输出

#### 3. 代码执行流程

1. 设置计算参数: 最大η值、步长、初始猜测值等
2. 调用`solve_blasius`函数求解Blasius方程
3. 计算摩擦系数和阻力
4. 将结果保存为Excel文件
5. 绘制速度剖面图像

#### 4. 结果分析

程序会计算以下关键参数:
- 不同η值对应的f、f'、f''值
- 壁面剪应力(τ_w)
- 局部摩擦系数(C_f)
- 平均摩擦系数(C_D)

这些结果会保存在Excel文件中，方便进一步分析和比较。

### 自定义参数

如果你需要修改计算参数，可以直接编辑`ode.py`中的以下部分:

```python
# 主要计算参数
eta_max = 10.0        # 最大η值
d_eta = 0.01          # η步长
f_double_prime_guess = 0.4696  # 初始猜测值，通常约为0.4696

# 流体和几何参数
rho = 1.225           # 流体密度(kg/m^3)
U_inf = 10.0          # 来流速度(m/s)
L = 1.0               # 平板长度(m)
```

调整这些参数可以模拟不同条件下的边界层流动。

### 常见问题

1. **计算不收敛**: 尝试调整`f_double_prime_guess`初始值或减小`d_eta`步长
2. **Excel文件无法打开**: 确保安装了pandas库，或手动检查输出路径是否正确
3. **图像显示异常**: 检查matplotlib配置，或尝试更新相关库版本

如果遇到其他问题，可以参考代码中的注释或提交Issue。
