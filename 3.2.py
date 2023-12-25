import numpy as np
import matplotlib.pyplot as plt

def PROGONKA(A, B):  #метод прогонки
     u = np.zeros_like(B)
     v = np.zeros_like(B)
     x = np.zeros_like(B)
     n = len(B)
     v[0] = A[0, 1] / (-A[0, 0])
     u[0] = (-B[0]) / (-A[0, 0])

     for i in range(len(B)-1):
          v[i] = -A[i, i + 1] / (A[i, i] + A[i, i - 1] * v[i - 1])
          u[i] = (-A[i, i - 1] * u[i - 1] + B[i]) / (A[i, i] + A[i, i - 1] * v[i - 1])

     v[n-1] = 0;
     u[n-1] = (A[n-1, n - 2] * u[n - 2] - B[n - 1]) / (-A[n - 1, n - 1] - A[n - 1, n - 2] * v[n - 2])
     x[n - 1] = u[n - 1];
     for i in range(n-2, -1, -1):
          x[i] = v[i] * x[i + 1] + u[i]

     return x

def H_func(X):
     H = np.zeros_like(X)
     for i in range(1 , len(X)):
          H[i] = X[i] - X[i-1]
     return H

def C_func(X, F, H): #вычисление коэффициентов c_i
     C = np.zeros(4)
     C[0] = 0.0
     len_A = 3
     A = np.zeros((len_A, len_A))
     A[0][0] = 2*(H[1]+H[2])
     A[0][1] = H[2]
     A[0][2] = 0
     B = np.zeros(len_A)
     B[0] = 3*((F[2]-F[1])/H[2] - (F[1]-F[0])/H[1])

     A[1][0] = H[2]
     A[1][1] = 2*(H[2]+H[3])
     A[1][2] = H[3]
     B[1] = 3*((F[3]-F[2])/H[3] - (F[2] - F[1])/H[2])

     A[2][0] = 0
     A[2][1] = H[3]
     A[2][2] = 2*(H[3] + H[4])
     B[2] = 3*((F[4] - F[3])/H[4] - (F[3]-F[2])/H[3])

     X = PROGONKA(A, B)
     for i in range(1, len(C)):
          C[i] = X[i - 1]

     return C

def A_func(F): #вычисление коэффициентов a_i
     A = np.zeros(len(F)-1)
     for i in range(0, len(A)):
          A[i] = F[i]
     return A

def B_func(F, H, C): #вычисление коэффициентов b_i
     B = np.zeros_like(C)
     for i in range(len(B) - 1):
          B[i] = (F[i + 1] - F[i]) / H[i + 1] - (1/3) * H[i + 1]*(C[i + 1] + 2*C[i])
     B[-1] = (F[-1] - F[-2]) / H[-1] - (2/3) * (H[-1]*C[-1])
     return B

def D_func(C, H): #вычисление коэффициентов d_i
     D = np.zeros_like(C)
     for i in range(len(C) - 1):
          D[i] = (C[i+1] - C[i])/(3*H[i+1])
     D[-1] = -C[-1]/(3*H[-1])
     return D


def spline(a, b, c, d, x_i, x): #функция сплайна
     F_x = a + b*(x - x_i) + c*(x-x_i)**2 + d*(x - x_i)**3
     return F_x

#задание
X = np.array([0.1, 0.5, 0.9, 1.3, 1.7])
F = np.array([-2.3026, -0.69315, -0.10536, 0.26236, 0.53063])

#коэффициенты для сплайна
H = H_func(X)
C = C_func(X, F, H)
print(f"C:{C}")
A = A_func(F)
print(f"A:{A}")
B = B_func(F, H, C)
print(f"B:{B}")
D = D_func(C, H)
print(f"D:{D}")

#вычисление функции в точке по заданию
X_ = 0.8
Y_ = spline(A[1], B[1], C[1], D[1], X[1], X_)

#визуализация
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
x_1 = np.linspace(X[0], X[1], 100)
x_2 = np.linspace(X[1], X[2], 100)
x_3 = np.linspace(X[2], X[3], 100)
x_4 = np.linspace(X[3], X[4], 100)

y_1 = np.array([spline(A[0], B[0], C[0], D[0], X[0], x) for x in x_1])
y_2 = np.array([spline(A[1], B[1], C[1], D[1], X[1], x) for x in x_2])
y_3 = np.array([spline(A[2], B[2], C[2], D[2], X[2], x) for x in x_3])
y_4 = np.array([spline(A[3], B[3], C[3], D[3], X[3], x) for x in x_4])
plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
plt.plot(x_3, y_3)
plt.plot(x_4, y_4)
plt.plot(X[0], F[0], 'ro')
plt.plot(X[1], F[1], 'ro')
plt.plot(X[2], F[2], 'ro')
plt.plot(X[3], F[3], 'ro')
plt.plot(X[4], F[4], 'ro')
plt.plot(X_, Y_, 'bo')
plt.grid()
plt.title(f"Значение функции в X*={X_}  {Y_}");
plt.show()


