# from autograd import elementwise_grad
# import autograd.numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from optimizers import *
from objectives import *
import numpy as np
from itertools import product

def create_power_function(degree=4, coeff=None, variance=1):
    return PowerFunction2D(degree=degree, coeff=coeff), Noise2D(degree=degree, variance=variance)
    # return PowerFunctionShifted(degree=degree, shift=np.array([1., .5]))

def create_psi_function():
    return Psi(B=7, b=3)
    # return PowerFunctionShifted(degree=degree, shift=np.array([1., .5]))


def plot_f1(func, ax):
    x = np.linspace(-1, 4, 500)
    y = np.linspace(-4, 1, 500)

    X, Y = np.meshgrid(x, y)
    a = np.reshape(X, (500*500, 1))
    b = np.reshape(Y, (500*500, 1))
    Z_ = np.concatenate((a, b), axis=1)
    # print(Z_.shape)
    # Z = func.f(X, Y)
    # print(Z_)
    Z = func.f(np_.transpose(Z_))
    # Z = func.f(Z_)
    # print(Z)
    Z = np.reshape(Z, (500, 500))
    # fig, ax = plt.subplots()
    ax.contour(X, Y, Z, 40, colors='black', linewidths=0.5)


def plot_x(x_final, x_history, ax, color):
    ax.scatter([i[0] for i in x_history], [i[1] for i in x_history], c=color, s=5, alpha=0.1)
    ax.scatter(x_final[0], x_final[1], c=color, s=5)
    # plt.show()

def plot_logf(x_history, ax, func, color, label, alpha=1):
    f_history = [func.f(x) for x in x_history]
    f_history = np.log(f_history)
    ax.plot(f_history, c=color, label=label, alpha=alpha)
    # plt.show()


def plot_s_vs_ns(variance, epsilon):

    func, noise = create_power_function(degree=6, coeff=np.array([[1., 1.], [1./2, -1./2]]), variance=variance)

    optim_s = StochasticExplicitMethod1(function=func, noise=noise, epsilon=epsilon, gamma=1, start_point=np.array([2., 1]))
    optim_ns = ExplicitMethod1(function=func, epsilon=epsilon, gamma=1, start_point=np.array([2., 1]))
    nsteps = 10000
    use_steps = True
    tolerance = 1e-6
    x_final_s, x_history_s = optim_s.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    x_final_ns, x_history_ns = optim_ns.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    print('Stochastic steps:', len(x_history_s))
    print('Non-stochastic steps:', len(x_history_ns))
    print(x_final_s, np.linalg.norm(x_final_s))
    print(x_final_ns, np.linalg.norm(x_final_ns))

    # fig, [ax1, ax2] = plt.subplots(1, 2)
    fig, ax2 = plt.subplots()
    # plot_f1(func, ax1)
    # plot_x(x_final_ns, x_history_ns, ax1, color='b')
    # plot_x(x_final_s, x_history_s, ax1, color='c')

    plot_logf(x_history_ns, ax2, func, color='b', label="Non-Stochastic")
    plot_logf(x_history_s, ax2, func, color='c', label="Stochastic")

    ax2.legend(loc='best')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('logf')

    plt.savefig('plots/s_vs_ns/s_vs_ns_var{}_eps{}.png'.format(variance, epsilon))


def plot_ns2_vs_gd_vs_mom(dim, mag='1'):

    COLORS = ['b', 'c', 'magenta', 'violet']#, 'purple', 'navy']
    fig, ax = plt.subplots()

    if mag == '1':
        mag = np.power(1./dim, 1./2)
    elif mag == 'tail':
        mag = 2
    elif mag == 'head':
        mag = 0.01


    func = Psi(B=8, b=2)

    # optim_ns = ExplicitMethod2(function=func, epsilon=0.01, gamma=0.9, start_point=np.array([0.01]*dim))
    optim_ns1 = ExplicitMethod1(function=func, epsilon=0.003, gamma=0.9, start_point=np.array([mag]*dim))
    optim_ns2 = ExplicitMethod2(function=func, epsilon=0.003, gamma=0.9, start_point=np.array([mag]*dim))
    optim_mom = Momentum(function=func, epsilon=0.003, gamma=0.9, start_point=np.array([mag]*dim))
    optim_gd = GradientDescent(function=func, epsilon=0.003, gamma=0.9, start_point=np.array([mag]*dim))
    
    nsteps = 10000
    use_steps = True
    tolerance = 1e-8

    x_final_ns1, x_history_ns1 = optim_ns1.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    print('Non-stochastic steps ns1:', len(x_history_ns1))
    x_final_ns2, x_history_ns2 = optim_ns2.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    print('Non-stochastic steps ns2:', len(x_history_ns2))
    x_final_mom, x_history_mom = optim_mom.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    print('Non-stochastic steps mom:', len(x_history_mom))
    x_final_gd, x_history_gd = optim_gd.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
    print('Non-stochastic steps gd:', len(x_history_gd))


    plot_logf(x_history_ns1, ax, func, color=COLORS[0], label='Explicit Method 1', alpha=0.5)
    plot_logf(x_history_ns2, ax, func, color=COLORS[1], label='Explicit Method 2', alpha=0.5)
    plot_logf(x_history_mom, ax, func, color=COLORS[2], label='Momentum', alpha=0.5)
    plot_logf(x_history_gd, ax, func, color=COLORS[3], label='Gradient Descent', alpha=0.5)
    

    ax.legend(loc='best')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('logf')

    plt.savefig('plots/ns2_vs_gd_vs_mom/comparison_dim{}_mag{}.png'.format(dim, mag))



def plot_ns1_gamma():

    GAMMA = [0.01, 0.1, 0.4, 0.7, 0.9, 0.99]
    COLORS = ['b', 'c', 'magenta', 'violet', 'purple', 'navy']
    fig, ax = plt.subplots()

    for gamma, color in zip(GAMMA, COLORS):

        func = PowerFunction2D(degree=4, coeff=np.array([[1., 1.], [1./2, -1./2]]))

        optim_ns = ExplicitMethod1(function=func, epsilon=0.007, gamma=gamma, start_point=np.array([2., 1]))
        
        nsteps = 10000
        use_steps = True
        tolerance = 1e-6

        x_final_ns, x_history_ns = optim_ns.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
        print('Non-stochastic steps:', len(x_history_ns))


        plot_logf(x_history_ns, ax, func, color=color, label='Gamma={}'.format(gamma), alpha=0.5)
    

    ax.legend(loc='best')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('logf')

    plt.savefig('plots/ns1_gamma/gamma.png'.format(gamma))


def plot_ns2():

    DIM = [4, 8, 16, 32, 64, 128]
    COLORS = ['b', 'c', 'magenta', 'violet', 'purple', 'navy']
    fig, ax = plt.subplots()

    for dim, color in zip(DIM, COLORS):

        func = Psi(B=8, b=2)

        optim_ns = ExplicitMethod2(function=func, epsilon=0.01, gamma=0.9, start_point=np.array([np.power(1./dim, 1./2)]*dim))
        
        nsteps = 10000
        use_steps = True
        tolerance = 1e-8

        x_final_ns, x_history_ns = optim_ns.optimize(steps=nsteps, tolerance=tolerance, use_steps=use_steps)
        print('Non-stochastic steps:', len(x_history_ns))


        plot_logf(x_history_ns, ax, func, color=color, label='dim={}'.format(dim), alpha=0.5)
    

    ax.legend(loc='best')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('logf')

    plt.savefig('plots/ns2_gamma/dim_11.png')



if __name__ == '__main__':

    # # func, noise = create_power_function(degree=4, coeff=np.array([[1., 1.], [1./2, -1./2]]), variance=0.6)
    # func = create_psi_function()
    # fig, ax = plot_f1(func)

    # optim = GradientDescent(function=func, epsilon=0.03, gamma=1, start_point=np.array([2., 1.]))
    # x_final_gd, x_history_gd = optim.optimize(steps=17000, tolerance=1e-6)
    # print('GD steps:', len(x_history_gd))

    # # optim = StochasticExplicitMethod1(function=func, noise=noise, epsilon=0.01, gamma=1, start_point=np.array([2., 0.5]))
    # optim = ExplicitMethod2(function=func, epsilon=0.03, gamma=0.99, start_point=np.array([2., 1]))
    # x_final_em1, x_history_em1 = optim.optimize(steps=100, tolerance=1e-6)
    # print('1st explicit method steps:', len(x_history_em1))

    # # plot_x(x_final=x_final_gd, x_history=x_history_gd, fig=fig, ax=ax)
    # plot_x(x_final=x_final_em1, x_history=x_history_em1, fig=fig, ax=ax)
    # plot_logf(x_history_gd=x_history, x_history_em1=x_history_em1, func=func)

    # VAR = [0.1, 0.9, 1.0, 2.0]
    # EPS = [0.1, 0.03]
    # for var, eps in product(VAR, EPS):
    #     plot_s_vs_ns(variance=var, epsilon=eps)

    # GAMMA = [0.1, 0.7, 1., 10.]
    # for gamma in GAMMA:
    #     plot_ns1_gamma(gamma)
    # plot_ns2()
    for dim in [2, 4, 8, 16, 64]:
        plot_ns2_vs_gd_vs_mom(dim=dim, mag='1')
        plot_ns2_vs_gd_vs_mom(dim=dim, mag='tail')
        plot_ns2_vs_gd_vs_mom(dim=dim, mag='head')