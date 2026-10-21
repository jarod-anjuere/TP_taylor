#-----------------------------------
# 1.1. Première application
#-----------------------------------

import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.subheader("Objectif de l'app")
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
    for i in range(n+1) :
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

st.subheader("Graphique des approximations")
fig, ax = plt.subplots()
ax.plot(X,Y)
ax.plot(X,Y_taylor)
st.pyplot(fig)

# Question : 
# Pour n fixé, l'approximation est de moins en moins bonne lorsque l'on s'éloigne de 0
# Pour ln(1+x), l'approximation ne devient pas meilleure lorsque n augmente pour x > 1


#-----------------------------------
# 5.Qualité de l'approximation
#-----------------------------------


# 5.1. Taux de coïncidence


def taux_coincidence(Y, Y_taylor, seuil) :
    mask_coincidence = abs(Y-Y_taylor) < seuil
    return mask_coincidence.mean()


with st.sidebar :
    seuil = st.select_slider("Choisir le seuil de comparaison", [0.000001, 0.00001, 0.0001, 0.001, 0.01], value=0.001)

st.write(f"Proportion de l'intervalle où l'approximation à un distance de {seuil} de la fonction : {taux_coincidence(Y, Y_taylor, seuil)}")

# Question : la comparaison avec nan renvoie faux ce qui diminue le taux de coincidence pour ln(1+x)


# 5.2. Courbe d'erreur

with st.sidebar :
    afficher_erreur = st.checkbox("Afficher l'erreur")

if afficher_erreur :
    st.subheader("Erreur de l'approximation")
    fig2, ax2 = plt.subplots()
    ax2.plot(X,abs(Y-Y_taylor))
    plt.yscale('log')
    st.pyplot(fig2)


#-----------------------------------
# 6. Comparer plusieurs ordres
#-----------------------------------

if "ordres" not in st.session_state :
    st.session_state.ordres = []

with st.sidebar :
    ajout_ordre = st.button("Ajouter cet ordre à la comparaison")

if ajout_ordre and ordre not in st.session_state.ordres :
    st.session_state.ordres.append(ordre)

st.subheader("Affichage des ordres sélectionnés")
fig3, ax3 = plt.subplots()
ax3.plot(X,Y)
for n in st.session_state.ordres : 
    ax3.plot(X,taylor2(X,f,n))
st.pyplot(fig3)