import math
import numpy as np

def der_one(xi, yi, x):
    i = 0
    while xi[i + 1] < x - 1e-7:
        i += 1
    return (yi[i + 1] - yi[i]) / (xi[i + 1] - xi[i])


def next_n(cur_n, prev_n, ans_cur, ans_prev, alpha1, beta1, B, b): #генерация следующего n для метода стрельбы
    num1 = beta1 * der_one(ans_cur[0], ans_cur[1], b)
    num2 = beta1 * der_one(ans_prev[0], ans_prev[1], b)
    num3 = alpha1 * ans_prev[1][len(ans_prev[0]) - 1]
    num4 = alpha1 * ans_cur[1][len(ans_cur[0]) - 1] + num1 - B
    num5 = alpha1 * ans_cur[1][len(ans_cur[0]) - 1] + num1 - num3 - num2
    return cur_n - num4 * (cur_n - prev_n) / num5

def Euler(f, xa, xb, ya, y1a, h):
    n = int((xb - xa) / h)
    x = xa
    y = ya
    x_res = [x]
    y_res = [y]
    y1 = y1a
    for i in range(n):
        y1 += h * f(x, y, y1)
        y += h * y1
        x += h
        x_res.append(x)
        y_res.append(y)
    return x_res, y_res

def solve_pogr(x, y):
    y1 = np.zeros_like(y)
    for i in range(len(y)):
        k = x[i] - 3 + 1/(x[i] + 1)
        y1[i] = abs(y[i] - k)
    return y1


def funcc(x):
    y1 = np.zeros_like(x)
    for i in range(len(x)):
        y1[i] = x[i] - 3 + 1/(x[i] + 1)
    return y1

def Runge_Rombert(h1, h2, y1, y2, p):
    y = np.zeros_like(y2)
    for i in range(len(y2)):
        y[i] = abs(y2[i] - (y1[2*i] + (y1[2*i] - y2[i])/(2**p - 1)))
    return y

def shooting_method(a, b, h, eps, f, alpha0, alpha1, beta0, beta1, A, B): #метод стрельбы
    n_prev = 1
    n_cur = 0.8

    ans_prev = Euler(f, a, b, n_prev, (A - alpha0 * n_prev) / beta0, h)
    ans_cur = Euler(f, a, b, n_cur, (A - alpha0 * n_cur) / beta0, h)
    while abs(alpha1 * ans_cur[1][len(ans_cur[0]) - 1] + \
          beta1 * der_one(ans_cur[0], ans_cur[1], b) - B) > eps:
        n = next_n(n_cur, n_prev, ans_cur, ans_prev, alpha1, beta1, B, b)
        n_prev = n_cur
        n_cur = n
        ans_prev = ans_cur
        ans_cur = Euler(f, a, b, n_cur, (A - alpha0*n_cur) / beta0, h)
    return ans_cur

def tma(a, b, c, d, shape):  #вспомогательный метод для конечных разностей
    p = [-c[0] / b[0]]
    q = [d[0] / b[0]]
    x = [0] * (shape + 1)
    for i in range(1, shape):
        p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
        q.append((d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
    for i in reversed(range(shape)):
        x[i] = p[i] * x[i + 1] + q[i]
    return x[:-1]

def finite_difference_method(a1, b1, h, alpha_0, alpha_1, beta_0, beta_1, A, B): #метод конечных разностей
    x = [a1]
    a = []
    b = []
    c = []
    d = []
    n = round((b1 - a1) / h)
    a.append(0)
    b.append(-2 / (h * (2 - p(a1) * h)) + q(a1) * h /
             (2 - p(a1) * h) + alpha_0 / beta_0)
    c.append(2 / (h * (2 - p(a1) * h)))
    d.append(A / beta_0 + h * f(a1) / (2 - p(a1) * h))
    x.append(x[0] + h)
    for i in range(1, n):
        a.append(1 / h**2 - p(x[i]) / (2 * h))
        b.append(-2 / h**2 + q(x[i]))
        c.append(1 / h**2 + p(x[i]) / (2 * h))
        d.append(f(x[i]))
        x.append(x[i] + h)
    a.append(-2 / (h * (2 + p(x[n]) *  h)))
    b.append(2 / (h * (2 + p(x[n]) * h)) - q(x[n]) * h /
             (2 + p(x[n]) * h) + alpha_1 / beta_1)
    c.append(0)
    d.append(B / beta_1 - h * f(x[n]) / (2 + p(x[n]) * h))
    y = tma(a, b, c, d, len(a))
    return x, y


#вариант - 17
print("вариант 17")
func = lambda x, y, y_der: ((3-x)*y_der+ y)/(x**2 - 1)
true_f = lambda x: np.exp(x) * x**2

f = lambda x: 0
q = lambda x: -1/(x**2 - 1) #функция при y''
p = lambda x: (x-3)/(x**2 - 1) #функция при y'

h = 0.05 #шаг
#коэффициенты для варианта 5
'''
q = lambda x: -2*(1+(np.tan(x)**2))
p = lambda x: 0
a = np.pi/4
x = np.array([a])
a1 = np.pi/4
b1 = np.pi/3
alpha_0 = 0
alpha_1 = -1
beta_0 = 1
beta_1 = 1
A = 3+ np.pi/2
B = 3 + np.pi *(4 - np.sqrt(3))/3'''

#коэффициенты для варианта 17
a2 = 0
b2 = 1
alpha_2 = 0
alpha_3 = 1
beta_2=1
beta_3=1
A1=0
B1=-0.75
eps=0.01

x = np.arange(0, 1 +h, h)

y = finite_difference_method(a2, b2, h, alpha_2, alpha_3, beta_2, beta_3,A1, B1)
y2 = finite_difference_method(a2, b2, h/2, alpha_2, alpha_3, beta_2, beta_3,A1, B1)
print("x:", np.array(y[0]))
print("Конечно-разностный метод:", np.array(y[1]))

y_romb = Runge_Rombert(h/2, h, y2[1], y[1], 4)
print("-------------\nПогрешность по методу Рунге-Ромберта:", np.array(y_romb))
y_solv = solve_pogr(y[0], y[1])
print("Погрешность в сравнении с точным значением:", np.array(y_solv))

y = res1 = shooting_method(a2, b2, h, eps, func, alpha_2, alpha_3, beta_2, beta_3,A1, B1)
y2= shooting_method(a2, b2, h / 2, eps, func, alpha_2, alpha_3, beta_2, beta_3,A1, B1)
print("\n\nx:", np.array(y[0]))
print(f"\nМетод стрельбы:", np.array(y[1]))

y_romb = Runge_Rombert(h/2, h, y2[1], y[1], 2)
print("-------------\nПогрешность по методу Рунге-Ромберта:", np.array(y_romb))
y_solv = solve_pogr(y[0], y[1])
print("Погрешность в сравнении с точным значением:", np.array(y_solv))

print("\nИстинное значение y:")
y_funcc = funcc(y[0])
print(y_funcc)