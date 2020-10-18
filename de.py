import matplotlib.pyplot as plt
from numpy import exp, log, cos, sin, double
import numpy as np


class DE:
    """
    class which contain needed entities for any differential equation: x, y, y'
    """
    def y_p(self, x, y, y_prim):
        """
        :param x: corresponding x
        :param y: corresponding y
        :param y_prim: function of y_prime(x,y) in string format
        :return: result of y_prime(x,y)
        """
        return eval(y_prim)

    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        """
        self.x_init = x_init
        assert self.x_init > 0, "x should bigger than 0" + str(self.x_init)
        self.y_init = y_init
        self.X = X
        assert self.X > self.x_init, "X should bigger than x_int"
        self.n = n
        assert self.n > 0, "n should bigger than 0"
        self.h = (self.X - self.x_init) / self.n
        self.y_prime = y_prime
        self.C = 1 / self.x_init - 1 / (exp(self.y_init) * self.x_init ** 2)
        self.x_arr = np.arange(self.x_init, self.X + self.h, self.h)


class Analytical_Method(DE):
    """
    class for solving DE by using Analytical Method
    """
    def y(self, x, C, y_fun):
        """
        :param x: x
        :param C: const
        :param y_fun: function in string format
        :return: result of y(x,C)
        """
        return eval(y_fun)

    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9, y_func="log(1/(x-C*x**2))"):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        :param y_func: function of y(x,C) in string format
        """
        DE.__init__(self, y_prime, x_init, y_init, X, n)
        self.y_func = y_func
        self.y_arr = np.array([self.y_init])

    def solve(self):
        """
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(1, self.x_arr.size):
            self.y_arr = np.append(self.y_arr, self.y(self.x_arr[i], self.C, self.y_func))
        return self.y_arr


class Numerical_Method(DE):
    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        """
        DE.__init__(self, y_prime, x_init, y_init, X, n)
        self.y_arr = self.create_y_arr(self.y_init)

    def create_y_arr(self, y_init):
        return np.array([y_init])


class Euler_Method(Numerical_Method):
    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        """
        Numerical_Method.__init__(self, y_prime, x_init, y_init, X, n)

    def solve(self):
        """
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr,
                                   self.y_arr[i] + self.h * self.y_p(self.x_arr[i], self.y_arr[i], self.y_prime))
        return self.y_arr

    def solve_LTE(self, y_ex):
        """
        :param y_ex: array of exact solution's y
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr, y_ex[i] + self.h * self.y_p(self.x_arr[i], y_ex[i], self.y_prime))
        return self.y_arr


class Improved_Euler_Method(Numerical_Method):
    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        """
        Numerical_Method.__init__(self, y_prime, x_init, y_init, X, n)

    def IEM(self, x, y, h):
        return self.y_p(x, y, self.y_prime) + self.y_p(x + h, y + h * self.y_p(x, y, self.y_prime), self.y_prime)

    def solve(self):
        """
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr,
                                   self.y_arr[i] + self.h / 2 * self.IEM(self.x_arr[i], self.y_arr[i], self.h))
        return self.y_arr

    def solve_LTE(self, y_ex):
        """
        :param y_ex: array of exact solution's y
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr, y_ex[i] + self.h / 2 * self.IEM(self.x_arr[i], y_ex[i], self.h))
        return self.y_arr


class Runge_Kutta(Numerical_Method):
    def RK(self, x, y, h):
        k1 = self.y_p(x, y, self.y_prime)
        k2 = self.y_p(x + h / 2, y + h * k1 / 2, self.y_prime)
        k3 = self.y_p(x + h / 2, y + h * k2 / 2, self.y_prime)
        k4 = self.y_p(x + h, y + h * k3, self.y_prime)
        return (k1 + 2 * k2 + 2 * k3 + k4)

    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        """
        Numerical_Method.__init__(self, y_prime, x_init, y_init, X, n)

    def solve(self):
        """
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr,
                                   self.y_arr[i] + self.h / 6 * self.RK(self.x_arr[i], self.y_arr[i], self.h))
        return self.y_arr

    def solve_LTE(self, y_ex):
        """
        :param y_ex: array of exact solution's y
        :return: array of y corresponding to array of  x
        """
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size - 1):
            self.y_arr = np.append(self.y_arr, y_ex[i] + self.h / 6 * self.RK(self.x_arr[i], y_ex[i], self.h))
        return self.y_arr


class Grid(DE):
    def __init__(self, y_prime="exp(y)-2/x", x_init=1, y_init=-2, X=10, n=9, check="1111"):
        """
        :param y_prime: function in string format
        :param x_init: beginning x
        :param y_init: beginning y
        :param X: end x
        :param n: number of steps
        :param check: if 1, build plot, no otherwise. 1 - Euler method; 2 -Improved Euler method; 3 - Runge-Kutta; 4 - exact
        """
        DE.__init__(self, y_prime, x_init, y_init, X, n)
        self.check = check
        self.em = Euler_Method(y_prime, x_init, y_init, X, n)
        self.iem = Improved_Euler_Method(y_prime, x_init, y_init, X, n)
        self.rk = Runge_Kutta(y_prime, x_init, y_init, X, n)
        self.am = Analytical_Method(y_prime, x_init, y_init, X, n)

    def plot_functions(self):
        """
        :return: plot of functions
        """
        self.ys = np.array([])
        self.names = np.array([])
        if self.check[0] == "1":
            self.ys = np.append(self.ys, self.em.solve())
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1':
            self.ys = np.append(self.ys, self.iem.solve())
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1':
            self.ys = np.append(self.ys, self.rk.solve())
            self.names = np.append(self.names, "Runge Kutta")
        if self.check[3] == '1':
            self.ys = np.append(self.ys, self.am.solve())
            self.names = np.append(self.names, "Analytical Method")
        if self.names.size > 0: self.ys = np.array_split(self.ys, self.names.size)
        plt.title("graphs for n = {0}, a = {1}, b = {2}".format(self.n, self.x_init, self.X), fontsize=20,
                  color='black')
        for i in range(self.names.size):
            plt.plot(self.x_arr, self.ys[i], label=self.names[i])
        if self.names.size > 0: plt.legend()
        plt.grid()
        plt.savefig("plot_functions.png")
        return plt.show()

    def GTE(self, y_ap, y_ex):
        """
        :param y_ap: array of approximate y
        :param y_ex: array of exact y
        :return: abs of their difference
        """
        return abs(y_ap - y_ex)

    def LTE(self, y_ap, y_ex):
        """
        :param y_ap: array of approximate y
        :param y_ex: array of exact y
        :return: abs of their difference
        """
        return abs(y_ap - y_ex)

    def plot_GTE(self):
        """
        :return: plot of GTE for some methods
        """
        self.y_am = self.am.solve()
        self.ys_err = np.array([])
        self.names = np.array([])
        if self.check[0] == "1":
            self.ys_err = np.append(self.ys_err, self.GTE(self.em.solve(), self.y_am))
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1':
            self.ys_err = np.append(self.ys_err, self.GTE(self.iem.solve(), self.y_am))
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1':
            self.ys_err = np.append(self.ys_err, self.GTE(self.rk.solve(), self.y_am))
            self.names = np.append(self.names, "Runge Kutta")
        if self.names.size > 0: self.ys_err = np.array_split(self.ys_err, self.names.size)
        plt.title("GTE for n = {0}, a = {1}, b = {2}".format(self.n, self.x_init, self.X), fontsize=20, color='black')
        for i in range(self.names.size):
            plt.plot(self.x_arr, self.ys_err[i], label=self.names[i])
        if self.names.size > 0: plt.legend()
        plt.grid()
        plt.savefig("plot_GTE.png")
        return plt.show()

    def plot_LTE(self):
        """
        :return: plot of GTE for some methods
        """
        self.y_am = self.am.solve()
        self.ys_err = np.array([])
        self.names = np.array([])
        if self.check[0] == "1":
            self.ys_err = np.append(self.ys_err, self.LTE(self.em.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1':
            self.ys_err = np.append(self.ys_err, self.LTE(self.iem.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1':
            self.ys_err = np.append(self.ys_err, self.LTE(self.rk.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Runge Kutta")
        if self.names.size > 0: self.ys_err = np.array_split(self.ys_err, self.names.size)
        plt.title("LTE for n = {0}, a = {1}, b = {2}".format(self.n, self.x_init, self.X), fontsize=20, color='black')
        for i in range(self.names.size):
            plt.plot(self.x_arr, self.ys_err[i], label=self.names[i])
        if self.names.size > 0: plt.legend()
        plt.grid()
        plt.savefig("plot_LTE.png")
        return plt.show()

    def plot_max_GTE(self, N=30, step=1):
        """
        for each step in range between n and N make approximation and plot maximum for each step
        :param N: end n
        :param step: size of step
        :return: plot of max GTE
        """
        assert N > self.n, "H should be bigger then h"
        assert step > 0, "step should be bigger then 0"
        self.names = np.array([])
        if self.check[0] == "1":
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == "1":
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == "1":
            self.names = np.append(self.names, "Runge Kutta")
        self.max_errs = []
        self.n_arr = np.arange(self.n, N + step, step)
        for i in self.n_arr:
            temp = Grid(self.y_prime, self.x_init, self.y_init, self.X, i, self.check)
            self.y_am = temp.am.solve()
            if self.check[0] == "1":
                self.max_errs.append(max(self.GTE(temp.em.solve(), self.y_am)))
            if self.check[1] == "1":
                self.max_errs.append(max(self.GTE(temp.iem.solve(), self.y_am)))
            if self.check[2] == "1":
                self.max_errs.append(max(self.GTE(temp.rk.solve(), self.y_am)))
        if self.names.size > 0: self.max_errs = np.transpose(
            np.array_split(self.max_errs, self.max_errs.__len__() / self.names.size))
        plt.title("GTE for n = {0},N = {1}, \nstep ={4}, a = {2}, b = {3}".format(self.n, N, self.x_init, self.X, step),
                  fontsize=20, color='black')
        for i in range(self.names.size):
            plt.plot(self.n_arr, self.max_errs[i], label=self.names[i])
        if self.names.size > 0: plt.legend()
        plt.grid()
        plt.xlabel("step size")
        plt.ylabel("max error")
        plt.savefig("plot_max_GTE.png")
        return plt.show()

    def plot_max_LTE(self, N=30, step=1):
        """
        for each step in range between n and N make approximation and plot maximum for each step
        :param N: end n
        :param step: size of step
        :return: plot of max LTE
        """
        assert N > self.n, "H should be bigger then n"
        assert step > 0, "step should be bigger then 0"
        self.names = np.array([])
        if self.check[0] == "1":
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == "1":
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == "1":
            self.names = np.append(self.names, "Runge Kutta")
        self.max_errs = []
        self.n_arr = np.arange(self.n, N + step, step)
        for i in self.n_arr:
            temp = Grid(self.y_prime, self.x_init, self.y_init, self.X, i, self.check)
            self.y_am = temp.am.solve()
            if self.check[0] == "1":
                self.max_errs.append(max(self.LTE(temp.em.solve_LTE(self.y_am), self.y_am)))
            if self.check[1] == "1":
                self.max_errs.append(max(self.LTE(temp.iem.solve_LTE(self.y_am), self.y_am)))
            if self.check[2] == "1":
                self.max_errs.append(max(self.LTE(temp.rk.solve_LTE(self.y_am), self.y_am)))
        if self.names.size > 0: self.max_errs = np.transpose(
            np.array_split(self.max_errs, self.max_errs.__len__() / self.names.size))
        plt.title("LTE for n = {0},N = {1}, \nstep ={4}, a = {2}, b = {3}".format(self.n, N, self.x_init, self.X, step),
                  fontsize=20, color='black')
        for i in range(self.names.size):
            plt.plot(self.n_arr, self.max_errs[i], label=self.names[i])
        if self.names.size > 0: plt.legend()
        plt.grid()
        plt.xlabel("step size")
        plt.ylabel("max error")
        plt.savefig("plot_max_LTE.png")
        return plt.show()