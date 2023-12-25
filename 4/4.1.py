import numpy as np
import math


def func(x, y, z):
    return (1+2*math.tan(x)**2)*y


def solve_pogr(x, y): #функция вычисления погрешности в сравнении с точным значением
    y1 = np.zeros_like(y)
    for i in range(len(y)):
        k = 1.0/(math.cos(x[i])) + math.sin(x[i]) + x[i]/(math.cos(x[i]))
        y1[i] = abs(y[i] - k)
    return y1


def funcc(x): #функция вычисления точного значения y
    y1 = np.zeros_like(x)
    for i in range(len(x)):
        y1[i] = 1.0 / (math.cos(x[i])) + math.sin(x[i]) + x[i] / (math.cos(x[i]))
    return y1

def func2(x, y, z):
    return z
def eiler(x, f, g, y0, z0, h): #метод Эйлера
    z = z0
    y = np.zeros_like(x)
    y[0] = y0
    k = 0
    for k in range(1, len(x)):
        z += h*f(x[k-1], y[k-1], z)
        y[k] = y[k-1] + h*g(x[k-1], y[k-1], z)
    return y

def Runge_Kutta(x, f, g, y0, z0, h): #метод Рунге-Кутты
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    y[0] = y0
    z[0] = z0
    for i in range( len(z) - 1):
        K1 = h*g(x[i], y[i], z[i])
        L1 = h * f(x[i], y[i], z[i])
        K2 = h * g(x[i] + 0.5*h, y[i] + 0.5*K1 , z[i] +0.5*L1)
        L2 = h * f(x[i] + 0.5*h, y[i] + 0.5*K1 , z[i] +0.5*L1)
        K3 = h* g(x[i] + 0.5*h, y[i] + 0.5*K2 , z[i] +0.5*L2)
        L3 = h * f(x[i] + 0.5*h, y[i] + 0.5*K2 , z[i] +0.5*L2)
        K4 = h* g(x[i] + h, y[i] + K3 , z[i] +L3)
        L4 = h * f(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = 1/6 *(K1 + 2*K2 +2*K3 +K4)
        delta_z = 1/6 *(L1 +2*L2 +2*L3 +L4)
        y[i+1] = y[i] + delta_y
        z[i+1] = z[i] + delta_z
    return y, z


def Adams(x, f, g, y0, z0, h): #метод Адамса
    y_runge, z_runge = Runge_Kutta(x, f, g, y0, z0, h)
    y_r = y_runge[:len(y_runge)//2 + 1]
    z_r = z_runge[:len(y_runge)//2 + 1]
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    for i in range(len(y)//2):
        y[i] = y_r[i]
        z[i] = z_r[i]
    for i in range(len(y)//2, len(y)):
        k = z[i]
        z[i] = z[i - 1] + h * (55 * f(x[i - 1], y[i - 1], z[i - 1]) -
                          59 * f(x[i - 2], y[i - 2], z[i - 2]) +
                          37 * f(x[i - 3], y[i - 3], z[i - 3]) -
                          9 * f(x[i - 4], y[i - 4], z[i - 4])) / 24.0
        k = y[i]
        y[i] = y[i - 1] + h * (55 * g(x[i - 1], y[i - 1], z[i - 1]) -
                          59 * g(x[i - 2], y[i - 2], z[i - 2]) +
                          37 * g(x[i - 3], y[i - 3], z[i - 3]) -
                          9 * g(x[i - 4], y[i - 4], z[i - 4])) / 24.0

    return y


def Runge_Romberg(h1, h2, y1, y2, p): #метод Рунге-Ромберга
    norm = 0
    y = np.zeros_like(y2)
    for i in range(len(y2)):
        y[i] = abs(y2[i] - (y1[2*i] + (y1[2*i] - y2[i])/(2**p - 1)))

    return y


print("Введите шаг h:")
h = float(input())
x = np.arange(0, 1+h, h)
x2 = np.arange(0, 1+h/2, h/2)

y = eiler(x,func, func2, 1, 2, h )
y2 = eiler(x2, func, func2, 1, 2, h/2)
y_romb = Runge_Romberg(h/2, h, y2, y, 1)
print("x:",np.array(x))
print("Метод Эйлера:")
print("y:",np.array(y))
print("-------------\nПогрешность по методу Рунге-Ромберта:", np.array(y_romb))
y_solv = solve_pogr(x, y)
print("Погрешность в сравнении с точным значением:", np.array(y_solv))


y_r = Runge_Kutta(x, func, func2, 1, 2, h)
y2 = Runge_Kutta(x2, func, func2, 1, 2, h/2)
y_romb = Runge_Romberg(h/2, h, y2[0], y_r[0], 4)
print("\nx:",np.array(x))
print("Метод Рунге-Кутты:")
print("y:",np.array(y_r[0]))
print("-------------\nПогрешность по методу Рунге-Ромберта:", np.array(y_romb))
y_solv = solve_pogr(x, y_r[0])
print("Погрешность в сравнении с точным значением:", np.array(y_solv))

y_a = Adams(x, func, func2, 1, 2, h)
y2 = Adams(x2, func, func2, 1, 2, h/2)
y_romb = Runge_Romberg(h/2, h, y2, y_a, 4)
print("\nx:",np.array(x))
print("Метод Адамса:")
print("y:",np.array(y_a))
print("-------------\nПогрешность по методу Рунге-Ромберта:", np.array(y_romb))
y_solv = solve_pogr(x, y_a)
print("Погрешность в сравнении с точным значением:", np.array(y_solv))

y_func = funcc(x)
print("\nИстинное значение функции\ny_true:", np.array(y_func))