import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log, cos, sin
from numpy import exp
class DE:
    def y_p(self,x,y,y_prim):
        return eval(y_prim)
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1):
        self.x_init = x_init
        assert self.x_init > 0, "x should bigger than 0" + str(self.x_init)
        self.y_init = y_init
        self.X = X
        assert self.X > self.x_init, "X should bigger than x_int"
        self.h = h
        assert self.h > 0, "h should bigger than 0"
        self.y_prime = y_prime
        self.C = 1/self.x_init - 1/(exp(self.y_init)*self.x_init**2)
        self.x_arr = np.arange(self.x_init,self.X,self.h)

class Analytical_Method(DE):
    def y(self,x,C,y_fun):
        return eval(y_fun)
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1, y_func = "log(1/(x-C*x**2))"):
        DE.__init__(self,y_prime, x_init , y_init , X, h)
        self.y_func = y_func
        self.y_arr = np.array([self.y_init])
    def solve(self):
        self.y_arr = np.array([self.y_init])
        for i in range(1,self.x_arr.size):
            self.y_arr = np.append(self.y_arr, self.y(self.x_arr[i],self.C, self.y_func))
        return  self.y_arr

class Numerical_Method(DE):
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1):
        DE.__init__(self,y_prime, x_init , y_init , X, h)
        self.y_arr = np.array([self.y_init])

class Euler_Method(Numerical_Method):
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1):
        Numerical_Method.__init__(self,y_prime, x_init , y_init , X, h)
    def solve(self):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,self.y_arr[i] + self.h*self.y_p(self.x_arr[i], self.y_arr[i],self.y_prime))
        return self.y_arr
    def solve_LTE(self,y_ex):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,y_ex[i] + self.h*self.y_p(self.x_arr[i], y_ex[i],self.y_prime))
        return self.y_arr

class Improved_Euler_Method(Numerical_Method):
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1):
        Numerical_Method.__init__(self,y_prime, x_init , y_init , X, h)
    def IEM(self,x,y,h):
        return self.y_p(x,y,self.y_prime) + self.y_p(x+h,y + h*self.y_p(x,y,self.y_prime),self.y_prime)
    def solve(self):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,self.y_arr[i] + self.h/2*self.IEM(self.x_arr[i],self.y_arr[i],self.h))
        return self.y_arr
    def solve_LTE(self,y_ex):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,y_ex[i] + self.h/2*self.IEM(self.x_arr[i],y_ex[i],self.h))
        return self.y_arr

class Runge_Kutta(Numerical_Method):
    def RK(self,x,y,h):
        k1 = self.y_p(x,y,self.y_prime)
        k2 = self.y_p(x + h/2,y+h*k1/2,self.y_prime)
        k3 = self.y_p(x + h/2, y + h*k2/2,self.y_prime)
        k4 = self.y_p(x + h,y + h*k3,self.y_prime)
        return (k1 + 2*k2 + 2*k3 + k4)
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1):
        Numerical_Method.__init__(self,y_prime, x_init , y_init , X, h)
    def solve(self):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,self.y_arr[i] + self.h/6*self.RK(self.x_arr[i],self.y_arr[i],self.h))
        return self.y_arr
    def solve_LTE(self, y_ex):
        self.y_arr = np.array([self.y_init])
        for i in range(self.x_arr.size-1):
            self.y_arr = np.append(self.y_arr,y_ex[i] + self.h/6*self.RK(self.x_arr[i],y_ex[i],self.h))
        return self.y_arr

class Grid(DE):
    def __init__(self,y_prime = "exp(y)-2/x", x_init = 1 , y_init = -2, X = 10, h = 1, check = "1111"):
        DE.__init__(self,y_prime, x_init , y_init , X, h)
        self.check = check
        self.em = Euler_Method(y_prime, x_init,y_init,X,h)
        self.iem = Improved_Euler_Method(y_prime, x_init,y_init,X,h)
        self.rk = Runge_Kutta(y_prime, x_init,y_init,X,h)
        self.am = Analytical_Method(y_prime, x_init,y_init,X,h)

    def plot_functions(self):
        self.ys = np.array([])
        self.names = np.array([])
        if self.check[0] == "1" :
            self.ys = np.append(self.ys, self.em.solve())
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1' :
            self.ys = np.append(self.ys, self.iem.solve())
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1' :
            self.ys = np.append(self.ys, self.rk.solve())
            self.names = np.append(self.names, "Runge Kutta")
        if self.check[3] == '1' :
            self.ys = np.append(self.ys, self.am.solve())
            self.names = np.append(self.names, "Analytical Method")
        if self.names.size > 0 : self.ys = np.array_split(self.ys, self.names.size)
        plt.title("graphs for h = {0}, a = {1}, b = {2}".format(self.h, self.x_init, self.X), fontsize = 20,color = 'black')
        for i in range(self.names.size):
            plt.plot(self.x_arr,self.ys[i],label = self.names[i])
        if self.names.size > 0 : plt.legend()
        plt.grid()
        plt.savefig("plot_functions.png")
        return plt.show()

    def GTE(self, y_ap, y_ex):
        return abs(y_ap - y_ex)
    def LTE(self, y_ap, y_ex):
        return abs(y_ap - y_ex)
    def plot_GTE(self):
        self.y_am = self.am.solve()
        self.ys_err = np.array([])
        self.names = np.array([])
        if self.check[0] == "1" :
            self.ys_err = np.append(self.ys_err, self.GTE(self.em.solve(), self.y_am))
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1' :
            self.ys_err = np.append(self.ys_err, self.GTE(self.iem.solve(), self.y_am))
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1' :
            self.ys_err = np.append(self.ys_err, self.GTE(self.rk.solve(), self.y_am))
            self.names = np.append(self.names, "Runge Kutta")
        if self.check[3] == '1' :
            self.ys_err = np.append(self.ys_err, self.am.solve() - self.y_am)
            self.names = np.append(self.names, "Analytical Method")
        if self.names.size > 0 : self.ys_err = np.array_split(self.ys_err, self.names.size)
        plt.title("GTE for h = {0}, a = {1}, b = {2}".format(self.h, self.x_init, self.X), fontsize = 20,color = 'black')
        for i in range(self.names.size):
            plt.plot(self.x_arr,self.ys_err[i],label = self.names[i])
        if self.names.size > 0 : plt.legend()
        plt.grid()
        plt.savefig("plot_GTE.png")
        return plt.show()

    def plot_LTE(self):
        self.y_am = self.am.solve()
        self.ys_err = np.array([])
        self.names = np.array([])
        if self.check[0] == "1" :
            self.ys_err = np.append(self.ys_err, self.LTE(self.em.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == '1' :
            self.ys_err = np.append(self.ys_err, self.LTE(self.iem.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == '1' :
            self.ys_err = np.append(self.ys_err, self.LTE(self.rk.solve_LTE(self.y_am), self.y_am))
            self.names = np.append(self.names, "Runge Kutta")
        if self.check[3] == '1' :
            self.ys_err = np.append(self.ys_err, self.am.solve() - self.y_am)
            self.names = np.append(self.names, "Analytical Method")
        if self.names.size > 0 : self.ys_err = np.array_split(self.ys_err, self.names.size)
        plt.title("LTE for h = {0}, a = {1}, b = {2}".format(self.h, self.x_init, self.X), fontsize = 20,color = 'black')
        for i in range(self.names.size):
            plt.plot(self.x_arr,self.ys_err[i],label = self.names[i])
        if self.names.size > 0 : plt.legend()
        plt.grid()
        plt.savefig("plot_LTE.png")
        return  plt.show()

    def plot_max_GTE(self, H = 5, step = 0.1):
        assert H > self.h , "H should be bigger then h"
        assert H < self.X , "H should be less then X"
        assert step > 0, "step should be bigger then 0"
        self.names = np.array([])
        if self.check[0] == "1" :
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == "1" :
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == "1" :
            self.names = np.append(self.names, "Runge Kutta")
        self.max_errs = []
        self.h_arr = np.arange(self.h, H+step, step)
        for i in self.h_arr:
            temp = Grid(self.y_prime, self.x_init, self.y_init, self.X, i, self.check)
            self.y_am = temp.am.solve()
            if self.check[0] == "1" :
                self.max_errs.append(max(self.GTE(temp.em.solve(), self.y_am)))
            if self.check[1] == "1" :
                self.max_errs.append(max(self.GTE(temp.iem.solve(), self.y_am)))
            if self.check[2] == "1" :
                self.max_errs.append(max(self.GTE(temp.rk.solve(), self.y_am)))
        if self.names.size > 0 : self.max_errs = np.transpose(np.array_split(self.max_errs, self.max_errs.__len__()/self.names.size))
        plt.title("GTE for h = {0},H = {1}, \nstep ={4}, a = {2}, b = {3}".format(self.h, H, self.x_init, self.X,step), fontsize = 20,color = 'black')
        for i in range(self.names.size):
            plt.plot(self.h_arr, self.max_errs[i], label = self.names[i])
        if self.names.size > 0 : plt.legend()
        plt.grid()
        plt.xlabel("step size")
        plt.ylabel("max error")
        plt.savefig("plot_max_GTE.png")
        return plt.show()

    def plot_max_LTE(self, H = 5, step = 0.1):
        assert H > self.h , "H should be bigger then h"
        assert H < self.X , "H should be less then X"
        assert step > 0, "step should be bigger then 0"
        self.names = np.array([])
        if self.check[0] == "1" :
            self.names = np.append(self.names, "Euler Method")
        if self.check[1] == "1" :
            self.names = np.append(self.names, "Improved Euler Method")
        if self.check[2] == "1" :
            self.names = np.append(self.names, "Runge Kutta")
        self.max_errs = []
        self.h_arr = np.arange(self.h, H+step, step)
        for i in self.h_arr:
            temp = Grid(self.y_prime, self.x_init, self.y_init, self.X, i, self.check)
            self.y_am = temp.am.solve()
            if self.check[0] == "1" :
                self.max_errs.append(max(self.LTE(temp.em.solve_LTE(self.y_am), self.y_am)))
            if self.check[1] == "1" :
                self.max_errs.append(max(self.LTE(temp.iem.solve_LTE(self.y_am), self.y_am)))
            if self.check[2] == "1" :
                self.max_errs.append(max(self.LTE(temp.rk.solve_LTE(self.y_am), self.y_am)))
        if self.names.size > 0 : self.max_errs = np.transpose(np.array_split(self.max_errs, self.max_errs.__len__()/self.names.size))
        plt.title("LTE for h = {0},H = {1}, \nstep ={4}, a = {2}, b = {3}".format(self.h, H, self.x_init, self.X,step), fontsize = 20,color = 'black')
        for i in range(self.names.size):
            plt.plot(self.h_arr, self.max_errs[i], label = self.names[i])
        if self.names.size > 0 : plt.legend()
        plt.grid()
        plt.xlabel("step size")
        plt.ylabel("max error")
        plt.savefig("plot_max_LTE.png")
        return plt.show()
