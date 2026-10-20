#-----------------------------------
# 1.1. Première application
#-----------------------------------

import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.header("Objectif de l'app")
st.write("Implémenter la formule de Taylor avec une accumulation (taylor1), la généraliser (taylor2), puis mesurer la qualité de l'approximation.")


#-----------------------------------
# 1.2. Afficher une formule
#-----------------------------------


st.latex(r"P_{n}(x) = \sum_{k=0}^{n}\frac{f^{(k)}(0)}{k!}x^{k}")


#-----------------------------------
# 1.3. Modèle d'exécution
#-----------------------------------


#valeur = st.slider("Choisissez une valeur", 0, 10)
#st.write("Vous avez choisi :", valeur)
#valeur mise à jour à chaque fois que l'on déplace le slider donc code exécuté plusieurs fois


#-----------------------------------
# 4. L'application : widgets et graphique
#-----------------------------------


# 4.1. Préparer app.py


import streamlit as st
import autograd.numpy as np
from autograd import grad
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


def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2 * np.sin(X) + np.cos(X/4)


# 4.2. Catalogue de fonctions


FONCTIONS = {'sin(x)' : np.sin,
             'cos(x)' : np.cos,
             'exp(x)' : np.exp,
             'ln(x+1)' : (lambda x : np.log(x+1)),
             '2sin(x) + cos(x/4)' : custom}


# 4.3. Widgets dans la barre latérale


with st.sidebar :
    nom_f = st.selectbox("choisir la fonction", FONCTIONS)
    ordre = st.slider("Choisir l'ordre n", 0, 10, value=3)
    largeur = st.slider("Choisir la largeur de l'intervalle", 2., 20., value=12.6)


# 4.4. Calculs et tracé


f = FONCTIONS[nom_f]
n = 400
X = np.linspace(-largeur/2, largeur/2, n)
Y = f(X)
Y_taylor = taylor2(X,f,ordre)

st.pyplot(plt.plot(X,Y))
st.pyplot(plt.plot(X,Y_taylor))
