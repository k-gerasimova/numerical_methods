import math
import numpy

#вспомогательные функции для многочлена Лангранжа
def w(x, X):
    return(x-X[0])*(x-X[1])*(x-X[2])*(x-X[3])


def w_d(x, X, num):
    pr = 1
    for i in range(4):
        if(num != i):
            pr *= (x-X[i])
    return pr


def mnog(x, X, Y, W):
    k=(Y[0]/W[0])*(x - X[1])*(x - X[2])*(x - X[3]) + (Y[1]/W[1])*(x - X[0])*(x - X[2])*(x - X[3])+(Y[2]/W[2])*(x - X[0])*(x- X[1])*(x - X[3])+ (Y[3]/W[3])*(x - X[0])*(x - X[1])*(x - X[2])
    return k


def func(x): #функция по варианту
    return math.log(x)


def print_part(X, Y, W):
    if (Y/W != 0):
        print(f"({Y/W})(x - {X[1]})(x - {X[2]})(x - {X[3]})", end="")
        return 1
    else:
        return 0


def print_all(X_a, Y_a, W_a):
    print(f"L(x) = ", end="")
    if (print_part(X_a, Y_a[0], W_a[0])):
        print("+")

    if (print_part(X_a, Y_a[1], W_a[1])):
        print("+")

    if (print_part(X_a, Y_a[2], W_a[2])):
        print("+", end="")
    print_part(X_a, Y_a[3], W_a[3])




print("вариант 5")
print("\n-------------------------Многочлен Лагранжа-------------------------")
X_a=[0.2, 0.6, 1.0, 1.4]
Y_a = [func(X_a[0]), func(X_a[1]), func(X_a[2]), func(X_a[3])]
W_a = [w_d(X_a[0], X_a, 0), w_d(X_a[1], X_a, 1), w_d(X_a[2], X_a, 2), w_d(X_a[3], X_a, 3)]

print("A)")
print_all(X_a, Y_a, W_a)
X = 0.8 #точка для вычисления значения
y = func(X)
L = mnog(X, X_a, Y_a, W_a)
print(f"\nАбсолютная погрешность интерполяции в точке Х*:{math.fabs(L-y)}")


X_b = [0.2, 0.6, 1.0, 1.4]
Y_b = [func(X_b[0]), func(X_b[1]), func(X_b[2]), func(X_b[3])]
W_b = [w_d(X_b[0], X_b, 0), w_d(X_b[1], X_b, 1), w_d(X_b[2], X_b, 2), w_d(X_b[3], X_b, 3)]

print("B)")
print_all(X_b, Y_b, W_b)

X = 0.8
y = func(X)
L = mnog(X, X_b, Y_b, W_b)
print(f"\nАбсолютная погрешность интерполяции в точке Х*:{math.fabs(L-y)}")
print(f"Значение в точке Х*:",L )

#--------------------------------------------------------------------------------
print("\n-------------------------Многочлен Ньютона-------------------------\n")
def Newton(x, X):
    return F([X[0]]) + (F([X[0], X[1]]))*(x - X[0]) +  (F([X[0], X[1], X[2]]))*(x - X[0])*(x - X[1])+ (F(X))*(x - X[0])*(x - X[1])*(x - X[2])

def F(X):
    if (len(X) > 1):
        k = (F(X[0:len(X)-1]) -F(X[1::]))/(X[0] - X[-1])
        return k
    else:
        z = func(X[0])
        return z


X = [0.2, 0.6, 1.0, 1.4]
print(f"{F([X[0]])} + ({F([X[0], X[1]])})(x - {X[0]}) +"
      f"\n({F([X[0], X[1], X[2]])})(x - {X[0]})(x - {X[1]})+"
      f"\n({F(X)})(x - {X[0]})(x - {X[1]})(x - {X[2]})")

X_1 = 0.8 #точка, в которой вычисляется значение функции
N = Newton(X_1, X)
print(f"Абсолютная погрешность интерполяции в точке Х*:{math.fabs(N-y)}")
print(f"Значение в точке Х*:", N )
