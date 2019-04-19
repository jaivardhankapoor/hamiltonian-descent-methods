from autograd import elementwise_grad
import autograd.numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

class ExplicitMethod1:
    def __init__(self, function, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.zeros_like(self.x).astype(np.complex128)

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        while (1):
            i += 1
            p_ = self.delta * self.p - self.epsilon * self.delta * (self.function.grad_f(self.x))
            self.p = p_
            x_ = self.x + self.epsilon * self.function.grad_k(self.p)
            if np.all(tolerance > abs(self.x - x_)):
                self.x = x_
                break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history


class StochasticExplicitMethod1:
    def __init__(self, function, noise, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.noise = noise
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.zeros_like(self.x).astype(np.complex128)

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        while (1):
            i += 1
            p_ = self.delta * self.p - self.epsilon * self.delta * (self.function.grad_f(self.x) + self.function.noise_f(self.x))
            self.p = p_
            x_ = self.x + self.epsilon * (self.function.grad_k(self.p) + self.function.noise_k(self.p))
            if np.all(tolerance > abs(self.x - x_)):
                self.x = x_
                break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history


class ExplicitMethod2:
    def __init__(self, function, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.zeros_like(self.x).astype(np.complex128)

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        j = 0
        while (1):
            i += 1
            x_ = self.x + self.epsilon * self.function.grad_k(self.p)
            p_ = (1 - self.epsilon * self.gamma) * self.p - self.epsilon * self.function.grad_f(x_)
            self.p = p_
            if np.all(tolerance > abs(self.x - x_)):
                j += 1
                self.x = x_
                if j > 1:
                    break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history

class StochasticExplicitMethod2:
    def __init__(self, function, noise, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.noise = noise
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.zeros_like(self.x).astype(np.complex128)

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        j = 0
        while (1):
            i += 1
            x_ = self.x + self.epsilon * (self.function.grad_k(self.p) + self.noise,noise_k(self.p))
            p_ = (1 - self.epsilon * self.gamma) * self.p - self.epsilon * (self.function.grad_f(x_) + self.noise.noise_f(x_))
            self.p = p_
            if np.all(tolerance > abs(self.x - x_)):
                j += 1
                self.x = x_
                if j > 1:
                    break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history

class Momentum:
    def __init__(self, function, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.array([0+0j, 0+0j])

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        # grad_f = elementwise_grad(self.function.f)
        # grad_k = elementwise_grad(self.function.k)
        while (1 and i < steps):
            print(i)
            print(self.x, self.p)
            i += 1
            p_ = self.delta * self.p - self.epsilon * self.delta * (self.function.grad_f(self.x))
            self.p = p_
            x_ = self.x + self.epsilon * self.function.grad_k_momentum(self.p)
            if np.all(tolerance > abs(self.x - x_)):
                self.x = x_
                break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history



class GradientDescent:
    def __init__(self, function, epsilon=0.1, gamma=1, start_point=None, dims=None):
        self.function = function
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = 1./(1 + self.gamma*self.epsilon)
        self.start_point = start_point
        self.dims = dims
        self.x = self.start_point
        if self.start_point is None:
            self.x = np.zeros(self.dims)
        else:
            self.dims = np.shape(self.start_point)[0]
        self.p = np.array([0+0j, 0+0j])

    def optimize(self, steps=100, tolerance=1e-6):
        i = 0
        x_history = []
        # grad_f = elementwise_grad(self.function.f)
        # grad_k = elementwise_grad(self.function.k)
        while (1):
            # print(i)
            # print(self.x, self.p)
            i += 1
            # p_ = self.delta * self.p - self.epsilon * self.delta * (self.function.grad_f(self.x))
            # self.p = p_
            x_ = self.x - self.epsilon * self.function.grad_f(self.x)
            if np.all(tolerance > abs(self.x - x_)):
                self.x = x_
                break
            self.x = x_
            x_history.append(self.x)

        return self.x, x_history

