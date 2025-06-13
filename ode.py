# 空气动力学作业：平板边界层（Blasius）方程的数值解法
# 简介：使用数值方法求解非线性常微分方程

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xlsxwriter
def blasius_ode(y, _):
    """
    Blasius equation in first-order form
    y[0] = f
    y[1] = f'
    y[2] = f''
    returns [f', f'', f''']
    """
    return [y[1], y[2], -0.5 * y[0] * y[2]]

def runge_kutta4(f, y0, t_span, h):
    """
    Fourth-order Runge-Kutta method for solving ODEs
    f: function defining the ODE
    y0: initial conditions
    t_span: [t_start, t_end]
    h: step size
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + h, h)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    for i in range(n - 1):
        k1 = np.array(f(y[i], t[i]))
        k2 = np.array(f(y[i] + k1 * h / 2, t[i] + h / 2))
        k3 = np.array(f(y[i] + k2 * h / 2, t[i] + h / 2))
        k4 = np.array(f(y[i] + k3 * h, t[i] + h))
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return t, y

def solve_blasius(eta_max=10.0, h=0.1):
    """
    Solve the Blasius boundary layer equation
    using the shooting method
    """
    # Initial guess for f''(0)
    f_double_prime_0 = 0.332  # Initial guess based on literature
    
    # Tolerance for convergence
    tol = 1e-6
    max_iter = 20
    
    for _ in range(max_iter):
        # Initial conditions [f(0), f'(0), f''(0)]
        y0 = [0, 0, f_double_prime_0]
        
        # Solve the ODE
        eta, solution = runge_kutta4(blasius_ode, y0, [0, eta_max], h)
        
        # Check convergence: f'(eta_max) should be close to 1
        f_prime_at_eta_max = solution[-1, 1]
        error = abs(f_prime_at_eta_max - 1.0)
        
        if error < tol:
            break
        
        # Update f''(0) using Newton-Raphson-like method
        f_double_prime_0 = f_double_prime_0 * (1.0 / f_prime_at_eta_max)
    
    # Extract the solution components
    f = solution[:, 0]
    f_prime = solution[:, 1]
    f_double_prime = solution[:, 2]
    
    return eta, f, f_prime, f_double_prime

def calculate_friction(eta, f_double_prime, Re_x):
    """
    Calculate wall shear stress, friction drag and friction coefficient
    """
    # Wall shear stress at η=0
    tau_w = f_double_prime[0]
    
    # Friction coefficient
    Cf = 0.664 / np.sqrt(Re_x)
    
    # Friction drag
    F_D = tau_w * np.sqrt(Re_x)
    
    return tau_w, F_D, Cf



# ... 已有代码 ...

def main():
    # Parameters
    eta_max = 10.0
    h = 0.1  # Step size as required in the problem
    
    # Solve the Blasius equation
    eta, f, f_prime, f_double_prime = solve_blasius(eta_max, h)
    
    # Calculate friction parameters (using a sample Reynolds number)
    Re_x = 1e5  # Example Reynolds number
    tau_w, F_D, Cf = calculate_friction(eta, f_double_prime, Re_x)
    
    # Create a table of results
    table_data = []
    print("\nResults Table (η from 0 to 10, step h=0.1):")
    print(f"{'η':>10} {'f':>15} {'f′':>15} {'f″':>15}")  # 使用Unicode字符
    print("-" * 55)
    
    for i in range(0, len(eta), 1):
        if eta[i] <= 10.0:  # Only include points up to eta=10
            print(f"{eta[i]:10.1f} {f[i]:15.6f} {f_prime[i]:15.6f} {f_double_prime[i]:15.6f}")
            table_data.append([eta[i], f[i], f_prime[i], f_double_prime[i]])
    
    # 确保 excel 文件夹存在
    excel_dir = 'excel'
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)
    
    workbook = xlsxwriter.Workbook(os.path.join(excel_dir, 'blasius_solution.xlsx'))
    
    # 第一个工作表 - 原始数据
    worksheet1 = workbook.add_worksheet('Original Data')
    headers1 = ['η', 'f', 'f′', 'f″']
    for col_num, header in enumerate(headers1):
        worksheet1.write(0, col_num, header)
    
    for row_num, row_data in enumerate(table_data, start=1):
        for col_num, value in enumerate(row_data):
            worksheet1.write(row_num, col_num, value)
    
    # 第二个工作表 - 摩擦参数
    worksheet2 = workbook.add_worksheet('Friction Parameters')
    headers2 = ['平板壁面摩擦应力 (τ_w)', '摩擦阻力 (F_D)', '摩擦系数 (Cf)']
    for col_num, header in enumerate(headers2):
        worksheet2.write(0, col_num, header)
    
    # 写入摩擦参数数据
    worksheet2.write(1, 0, tau_w)
    worksheet2.write(1, 1, F_D)
    worksheet2.write(1, 2, Cf)
    
    # 关闭工作簿
    workbook.close()

    # ... 已有代码 ...


    # 确保 fig 文件夹存在
    fig_dir = 'fig'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Plot velocity profiles
    plt.figure(figsize=(10, 8))
    
    # Plot f, f', f''
    ax1 = plt.subplot(111)
    ax1.plot(eta, f, 'b-', linewidth=2, label='$f$')
    ax1.plot(eta, f_prime, 'r-', linewidth=2, label='$f\'$')
    ax1.plot(eta, f_double_prime, 'g-', linewidth=2, label='$f\'\'$')
    
    ax1.set_xlabel('$\eta$', fontsize=14)
    ax1.set_ylabel('Functions', fontsize=14)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True)
    ax1.set_xlim([0, 10])
    ax1.xaxis.set_major_locator(MaxNLocator(11))  # 11 ticks from 0 to 10
    
    plt.title('Blasius Boundary Layer Solution', fontsize=16)
    plt.tight_layout()
    
    # 保存图片到 fig 文件夹
    plt.savefig(os.path.join(fig_dir, 'blasius_solution.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片窗口，避免显示

    # Print friction results
    print("\nFriction Parameters:")
    print(f"Wall shear stress (τ_w): {tau_w:.6f}")
    print(f"Friction drag (F_D): {F_D:.6f}")
    print(f"Friction coefficient (Cf): {Cf:.6f}")

if __name__ == "__main__":
    main()
