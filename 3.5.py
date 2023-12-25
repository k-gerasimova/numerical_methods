import numpy as np

def func(x): #функция y по заданию
    return 1/((2*x+7)*(3*x+4))


def pryamougolnic(X, H): #метод прямоугольников
    sum = 0
    for i in range(1,len(X)):
        sum += func((X[i-1] + X[i])/2)
    sum *= H
    return sum

def trap(Y, H): #метод трапеций
    sum = Y[0]/2 + Y[-1]/2
    for i in range(1, len(Y) - 1):
        sum += Y[i]
    sum *= H
    return sum

def Simp(Y, h): #метод Симпсона
    sum = Y[0] + Y[-1]
    for i in range(1, len(Y)-1):
        if (i % 2 == 0):
            sum += Y[i] * 2
        if (i % 2 != 0):
            sum += Y[i] * 4
    sum *= h/3
    return sum

#данные по заданию
X0 = -1
Xk = 1

#шаги для вычисления
h1 = 0.5
h2 = 0.25

x1 = np.arange(X0, Xk + h1, h1)
x2 = np.arange(X0, Xk + h2, h2)

y1 = np.array([func(x) for x in x1])
y2 = np.array([func(x) for x in x2])

#вычисления для шагов h1 и h2
pryam = pryamougolnic(x1, h1)
trapec = trap(y1, h1)
simpson = Simp(y1, h1)

pryam1 = pryamougolnic(x2, h2)
trapec1 = trap(y2, h2)
simpson1 = Simp(y2, h2)

print(f"Метод треугольников, значения интеграла с шагом h1=0.5: F={pryam}, h2=0.25: F={pryam1}")
print(f"Метод трапеций, значения интеграла с шагом h1=0.5: F={trapec}, h2=0.25: F={trapec1}")
print(f"Метод Симпсона, значения интеграла с шагом h1=0.5: F={simpson}, h2=0.25: F={simpson1}")

print("\nметод Рунге-Ромберга-Ричардсона:")
F = pryam1 + (pryam1 - pryam)/(2**2 - 1)
print(f"для метода треугольников:{F}")
F1 = trapec1 + (trapec1 - trapec)/(2**2 - 1)
print(f"для метода трапеций:{F1}")
F2 = simpson1 + (simpson1 - simpson)/(2**2 - 1)
print(f"для метода Симпсона:{F2}")

print("\nПогрешности:")
print(f"\nметод треугольников:\nдля шага 0.5: {abs(F-pryam)}\nдля шага 0.25: {abs(F-pryam1)}")
print(f"\nметод трапеций:\nдля шага 0.5: {abs(F1-trapec)}\nдля шага 0.25: {abs(F1-trapec1)}")
print(f"\nметод Симпсона:\nдля шага 0.5: {abs(F2-simpson)}\nдля шага 0.25: {abs(F2-simpson1)}")