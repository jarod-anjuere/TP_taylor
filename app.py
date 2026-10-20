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




