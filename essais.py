
#-----------------------------------
# 2. Découverte d'autograd
#-----------------------------------


import autograd.numpy as np
from autograd import grad

# Question 1
n = 100
X = np.linspace(0,2*np.pi,n)

# Question 2
sin_der = np.vectorize(grad(np.sin))
Y = sin_der(X)

# Question 3
import math as m
Y2 = np.cos(X)
close = np.vectorize(m.isclose)
print(close(Y,Y2).all())


#-----------------------------------
# 3. Formule de Taylor
#-----------------------------------


# 3.1. Version 1 : dérivées fournies


from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
    Y = np.zeros(len(X))
    for k in range(len(derivatives)) :
        Y = Y + derivatives[k]*(X**k)/factorial(k)
    return Y
    

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)
Y = taylor1(X, sinus10)
plt.plot(X,Y)
plt.plot(X,np.sin(X))
plt.show()


# 3.2. Version 2 : dérivées calculées avec autograd


def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
    derivatives = []
    d_f = f
    for i in range(n) :
        derivatives.append(d_f(0.0))
        d_f = grad(d_f)
    return taylor1(X,derivatives)


# tests

# exp à l'ordre 2 : 1 + x + x^2/2
print(np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2))
# renvoie False

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)

# Question 1
plt.plot(X,Y2)
plt.plot(X,Y)
plt.show()


# Question 2
mask_approx = abs(Y-Y2) < 0.001
print(mask_approx.sum()/len(X))
# On trouve 42% de l'intervalle


# Question 3
Y_cos = np.cos(X)
Y2_cos = taylor2(X, np.cos, 0)
plt.plot(X,Y_cos)
plt.plot(X,Y2_cos)
plt.show()
# approximation constante


# 3.3. Une fonction personnalisée


def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2 * np.sin(X) + np.cos(X/4)

Y3 = custom(X)
Y4 = taylor2(X, custom, 14)
plt.plot(X,Y3)
plt.plot(X,Y4)
plt.show()

mask_approx = abs(Y3-Y4) < 0.001
print(mask_approx.sum()/len(X))
# On trouve 61,5% de l'intervalle





