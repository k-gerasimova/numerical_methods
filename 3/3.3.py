import numpy as np
import matplotlib.pyplot as plt

def make_a1(X, Y): #вычисление коэффициента a1 для приближающего многочлена 1 степени
    sum_y = 0
    sum_x=0
    sum_x2=0
    sum_xy=0
    for i in range(len(Y)):
        sum_y += Y[i]
        sum_x += X[i]
        sum_x2 += X[i]**2
        sum_xy += X[i]*Y[i]

    return (sum_x*sum_y - sum_xy*(len(Y)))/(sum_x**2-((len(Y))*sum_x2))


def make_a0(X, Y, a1): #вычисление коэффициента a0 для приближающего многочлена 1 степени
    sum_y = 0
    sum_x = 0
    for i in range(len(Y)):
        sum_y += Y[i]
        sum_x += X[i]
    return (sum_y-a1*sum_x)/(len(Y))


def Mn_1(a0, a1, x): #построение приближающего многочлена 1 степени
    return a0 + a1*x

def Mn_2(a0, a1, a2, x): #построение приближающего многочлена 2 степени
    return a0 +a1*x+a2*x**2

X = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
Y = [-2.3026, -0.69315, -0.10536, 0.26236, 0.53063, 0.74194]

print("вариант 5:")
#коэффициенты для приближающего многочлена 1 степени
a1 = make_a1(X, Y)
print(f"a1:{a1}")
a0 = make_a0(X, Y, a1)
print(f"a0:{a0}")


#сумма квадратов ошибок многочлена 1 степени
Phi = 0
for i in range(len(Y)):
    Phi += (Mn_1(a0, a1, X[i]) - Y[i])**2
print(f"Phi mn 1:{Phi}")

sum_x=sum_y=sum_x2=sum_x3=sum_xy=sum_x4=sum_yx2=0
for i in range(len(Y)):
    sum_x += X[i]
    sum_y += Y[i]
    sum_x2 += X[i]**2
    sum_x3 += X[i]**3
    sum_x4 += X[i]**4
    sum_xy += X[i]*Y[i]
    sum_yx2 += Y[i]*X[i]**2
A = np.array([[len(Y), sum_x, sum_x2], [sum_x, sum_x2, sum_x3], [sum_x2, sum_x3, sum_x4]])
B = np.array([sum_y, sum_xy, sum_yx2])
a = np.linalg.solve(A, B)
#коэфициенты для приближающего многочлена второй степени
print(a)


#Сумма квадратов ошибок для многочлена второй степени
Phi = 0
for i in range(len(Y)):
    Phi += (Mn_2(a[0], a[1], a[2], X[i]) - Y[i])**2

print(f"Phi mn 2:{Phi}")

#визуализация
x = np.linspace(X[0], X[-1], 500)
y_1 = np.array([Mn_1(a0, a1, k) for k in x])
y_2 = np.array([Mn_2(a[0], a[1], a[2], k) for k in x])
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
plt.plot(X, Y, 'ro')
plt.plot(x, y_1, "g")
plt.plot(x, y_2, "b")
plt.grid()
plt.show()
