from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from optimizers import ExplicitMethod1, ExplicitMethod2, Momentum, GradientDescent
from autograd import elementwise_grad
import autograd.numpy as np
import numpy as np_

class Noise2D:
    def __init__(self, variance=1, degree=1):
        self.variance = variance
        self.degree = degree
    
    def noise_f(self, x):
        norm = (np.abs(x[0])**self.degree + np.abs(x[0])**self.degree)**(1./self.degree)
        return np.random.normal(0, norm * self.variance)
        
    def noise_f(self, x):
        degree_ = 1./(1 - 1./self.degree)
        norm = (np.abs(x[0])**degree_ + np.abs(x[0])**degree_)**(1./degree_)
        return np.random.normal(0, norm * self.variance)

class PowerFunction2D:
    '''
    Uses kinetic function suited for 1st explicit method.
    '''

    def __init__(self, degree=2, coeff=None):
        self.degree = degree
        self.coeff = coeff
    
    def f(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        return (x*self.coeff[0][0] + y*self.coeff[0][1]) ** self.degree \
               + (x*self.coeff[1][0] + y*self.coeff[1][1]) ** self.degree
    
    def k(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        degree_ = 1./(1 - 1/self.degree)
        return 1./degree_ * (x ** (degree_) + y ** degree_)

    def k_momentum(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        degree_ = 2
        return 1./degree_ * (x ** (degree_) + y ** degree_)

    def grad_f(self, x):
        _grad = np.array([0., 0.])
        _grad[0]  = self.degree * self.coeff[0][0] * (x[0]*self.coeff[0][0] + x[1] * self.coeff[0][1]) ** (self.degree - 1) \
                    + self.coeff[1][0] * self.degree * (x[0] * self.coeff[1][0] + x[1] * self.coeff[1][1]) ** (self.degree - 1)
        _grad[1]  = self.degree * self.coeff[0][1] * (x[0]*self.coeff[0][0] + x[1] * self.coeff[0][1]) ** (self.degree - 1) \
                    + self.coeff[1][1] * self.degree * (x[0] * self.coeff[1][0] + x[1] * self.coeff[1][1]) ** (self.degree - 1)
        return _grad
    
    def grad_k(self, x):
        degree_ = 1./(1 - 1/self.degree)
        _grad  =  np.real(np.sign(x)*np.abs([x[0] ** (degree_ - 1), x[1] ** (degree_ - 1)]))
        return _grad

    def grad_k_momentum(self, x):
        degree_ = 2
        _grad  =  np.real(np.sign(x)*np.abs([x[0] ** (degree_ - 1), x[1] ** (degree_ - 1)]))
        return _grad


class PowerFunctionShifted:

    def __init__(self, degree=2, shift=None):
        self.degree = degree
        self.shift = shift
    
    def f(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        return (x-self.shift[0]) ** self.degree \
               + (y-self.shift[1]) ** self.degree
    
    def k(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        degree_ = 1./(1 - 1/self.degree)
        return 1./degree_ * (x ** (degree_) + y ** degree_)

    def k_momentum(self, x, y=None):
        if y is None:
            y = x[1]
            x = x[0]
        degree_ = 2
        return 1./degree_ * (x ** (degree_) + y ** degree_)

    def grad_f(self, x):
        _grad = np.array([0., 0.])
        _grad[0]  = self.degree * (x[0]-self.shift[0]) ** (self.degree - 1)
        _grad[1]  = self.degree * (x[1]-self.shift[1]) ** (self.degree - 1)
        return _grad
    
    def grad_k(self, x):
        degree_ = 1./(1 - 1/self.degree)
        _grad  =  np.real(np.sign(x)*np.abs([x[0] ** (degree_ - 1), x[1] ** (degree_ - 1)]))
        return _grad

    def grad_k_momentum(self, x):
        degree_ = 2
        _grad  =  np.real(np.sign(x)*np.abs([x[0] ** (degree_ - 1), x[1] ** (degree_ - 1)]))
        return _grad



def create_power_function(degree=4, coeff=None):
    return PowerFunction2D(degree=degree, coeff=coeff)
    # return PowerFunctionShifted(degree=degree, shift=np.array([1., .5]))


def plot_f1(func):
    x = np.linspace(-1, 4, 500)
    y = np.linspace(-4, 1, 500)

    X, Y = np.meshgrid(x, y)
    a = np.reshape(X, (500*500, 1))
    b = np.reshape(Y, (500*500, 1))
    Z_ = np.concatenate((a, b), axis=1)
    # print(Z_.shape)
    # Z = func.f(X, Y)
    Z = func.f(np.transpose(Z_))
    Z = np.reshape(Z, (500, 500))
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, 40, colors='black', linewidths=0.5)
    return (fig, ax)


def plot_x(x_final, x_history, fig, ax):
    ax.scatter([i[0] for i in x_history], [i[1] for i in x_history], c='k', s=5)
    ax.scatter(x_final[0], x_final[1], c='b')
    plt.show()

if __name__ == '__main__':
    func = create_power_function(degree=4, coeff=np.array([[1., 1.], [1./2, -1./2]]))
    # func = create_power_function(degree=1.5, coeff=np.array([[1., 0], [1./2, 0]]))
    fig, ax = plot_f1(func)

    # i = 10000000000000
    # for gamma in np.linspace(0.1, 0.9, 9):
    #     optim = Momentum(function=func, epsilon=0.01, gamma=gamma, start_point=np.array([2., 1.]))
    #     x_final, x_history = optim.optimize(steps=17000, tolerance=1e-6)
    #     if i > len(x_history):
    #         i = len(x_history)
    # print(len(x_history))

    # i = 10000000000000
    # for gamma in np.linspace(0.1, 0.9, 9):
    #     optim = ExplicitMethod1(function=func, epsilon=0.01, gamma=gamma, start_point=np.array([2., 1.]))
    #     x_final, x_history = optim.optimize(steps=17000, tolerance=1e-6)
    #     if i > len(x_history):
    #         i = len(x_history)
    # print(len(x_history))


    # i = 10000000000000
    # optim = GradientDescent(function=func, epsilon=0.01, gamma=gamma, start_point=np.array([2., 1.]))
    # x_final, x_history = optim.optimize(steps=17000, tolerance=1e-6)
    # print(len(x_history))

    optim = ExplicitMethod2(function=func, epsilon=0.01, gamma=1, start_point=np.array([2., 0.5]))
    x_final, x_history = optim.optimize(steps=170, tolerance=1e-6)
    print(len(x_history))

    plot_x(x_final=x_final, x_history=x_history, fig=fig, ax=ax)