# TP - Approximation de Taylor avec autograd et Streamlit

Durée : 2 h.

## Contexte

On approche localement une fonction par un polynôme, plus simple à évaluer, dériver ou intégrer. Ici, développement en 0 :

$$
P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k
$$

Les dérivées successives sont calculées automatiquement avec **autograd**. L'application **Streamlit** finale permet de choisir une fonction et un ordre, de visualiser l'approximation, d'estimer l'erreur et de comparer plusieurs ordres.

## Objectifs

- Calculer des dérivées successives avec `autograd`.
- Implémenter la formule de Taylor avec une accumulation (`taylor1`), puis la généraliser (`taylor2`).
- Mesurer la qualité d'une approximation (taux de coïncidence à un seuil donné).
- Construire une interface Streamlit avec widgets, barre latérale et `st.session_state`.

## Prérequis

- Dérivées et formule de Taylor.
- Python : fonctions, boucles, listes. NumPy et Matplotlib (bases).
- Aucune connaissance de Streamlit requise.

## Installation

```bash
pip install streamlit autograd matplotlib
```

Cheatsheet : https://cheat-sheet.streamlit.app/
Documentation : https://docs.streamlit.io/library/api-reference

## Structure du projet

```text
tp_taylor/
├── essais.py   # tests hors Streamlit (parties 2 et 3)
└── app.py      # application (parties 1, 4 à 6)
```

Lancement : `streamlit run app.py`

---

## 1. Découverte de Streamlit

### 1.1. Première application

Créez `app.py` :

```python
import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
```

**Consigne :** ajoutez un sous-titre d'une phrase présentant l'objectif de l'application.

<details>
<summary><b>Indice</b></summary>

Streamlit propose des fonctions pour les titres de différents niveaux, comme `h1`, `h2` en HTML.

</details>

### 1.2. Afficher une formule

**Consigne :** affichez la formule de $P_n$ avec `st.latex`.

### 1.3. Modèle d'exécution

Ajoutez temporairement :

```python
valeur = st.slider("Choisissez une valeur", 0, 10)
st.write("Vous avez choisi :", valeur)
```

**Question :** quand vous déplacez le slider, le script est-il exécuté une seule fois ou plusieurs fois ?

Supprimez ce code de test.

---

## 2. Découverte d'autograd

À partir d'ici, dans `essais.py` (hors Streamlit).

```python
# À LA PLACE de l'habituel 'import numpy as np'
import autograd.numpy as np
from autograd import grad
```

- `autograd.numpy` expose les mêmes fonctions que NumPy, mais dérivables. `np` désigne désormais ce module.
- `grad(f)` retourne la fonction dérivée de `f`. Elle ne s'applique qu'à un **scalaire**.

### 2.1. Domaine

**Q1 :** calculez un tableau `X` de réels entre $0$ et $2\pi$.

### 2.2. Dérivée

**Q2 :** définissez `sin_der`, la fonction dérivée de `sin`, puis appliquez-la à `X`.

<details>
<summary><b>Indice</b></summary>

`grad` ne sait traiter que des scalaires. Pour appliquer une fonction sur un tableau, il faut la vectoriser (`np.vectorize`).

</details>

### 2.3. Vérification

**Q3 :** vérifiez que `sin_der(X)` est bien égal à `np.cos(X)`.

---

## 3. Formule de Taylor

### 3.1. Version 1 : dérivées fournies

**Consigne :** écrivez

```python
from math import factorial

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
```

Testez avec les dérivées de sinus en 0, passées en dur :

```python
sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)
```

Tracez `taylor1(X, sinus10)` avec Matplotlib et comparez avec `np.sin(X)`.

<details>
<summary><b>Indice</b></summary>

Boucle sur les dérivées en accumulant dans `Y` le terme $f^{(i)}(0) \, x^i / i!$. Utilisez `factorial` du module `math`.

</details>

### 3.2. Version 2 : dérivées calculées avec autograd

**Consigne :** écrivez

```python
def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
```

<details>
<summary><b>Indice</b></summary>

Construisez la liste des $f^{(k)}(0)$ pour $k$ de $0$ à $n$ en dérivant `f` à chaque tour de boucle avec `grad`, puis réutilisez `taylor1`. Évaluez en `0.0` (flottant).

</details>

**Tests :**

```python
# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)
```

**Questions :**

1. Tracez `Y` et `Y2` sur le même graphique.
2. Sur quel pourcentage du domaine l'approximation est-elle valide à $10^{-3}$ près ?
3. Tracez cosinus et son approximation au degré 0. Que constatez-vous ?

### 3.3. Une fonction personnalisée

**Consigne :** complétez

```python
def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
```

puis calculez `Y3 = custom(X)` et `Y4 = taylor2(X, custom, 14)`. Comparez-les graphiquement et donnez le pourcentage du domaine où la coïncidence est à $10^{-3}$ près.

---

## 4. L'application : widgets et graphique

On rapatrie les fonctions dans `app.py` et on construit l'interface.

### 4.1. Préparer `app.py`

Copiez `taylor1`, `taylor2` et `custom` en haut de `app.py`, avec les imports :

```python
import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt
```

### 4.2. Catalogue de fonctions

**Consigne :** définissez un dictionnaire `FONCTIONS` associant un nom à une fonction scalaire, pour : $\sin(x)$, $\cos(x)$, $e^x$, $\ln(1+x)$ et la fonction `custom`.

<details>
<summary><b>Indice</b></summary>

Pour $\ln(1+x)$, utilisez une `lambda`.

</details>

### 4.3. Widgets dans la barre latérale

**Consigne :** dans la barre latérale, ajoutez :

- une `selectbox` pour choisir la fonction ;
- un `slider` pour l'ordre `n` (de $0$ à $10$, défaut $3$) ;
- un `slider` pour la largeur de l'intervalle étudié, `largeur` (de $2$ à $20$, défaut $12.6$, soit environ $4\pi$).

<details>
<summary><b>Indice</b></summary>

Un conteneur dédié à la barre latérale s'utilise avec `with st.sidebar:`.

</details>

### 4.4. Calculs et tracé

**Consigne :** dans la zone principale :

1. récupérez `f` dans `FONCTIONS` ;
2. construisez `X` de 400 points sur $[-\text{largeur}/2, \text{largeur}/2]$ ;
3. calculez `Y` (valeurs de `f`) et `Y_taylor` (approximation) ;
4. tracez les deux courbes avec légende, et affichez avec `st.pyplot`.

<details>
<summary><b>Indice</b></summary>

`f` ne s'applique qu'à un scalaire dans `taylor2` (via `grad`), mais sur un tableau pour calculer `Y` : `np.sin`, `np.exp`, etc. et `custom` sont déjà vectorisées pour cela.

</details>

**Attention :** $\ln(1+x)$ n'est pas définie pour $x \le -1$ : les valeurs `nan` correspondantes sont ignorées par Matplotlib.

**Question :** pour $n$ fixé, comment évolue la qualité de l'approximation quand on s'éloigne de $0$ ? Que se passe-t-il pour $\ln(1+x)$ au-delà de $|x| = 1$ quand on augmente $n$ ?

---

## 5. Qualité de l'approximation

### 5.1. Taux de coïncidence

**Consigne :** écrivez `taux_coincidence(Y, Y_taylor, seuil)`, qui retourne la proportion de l'intervalle où $|Y - Y_{taylor}|$ est inférieur au seuil. Ajoutez dans la barre latérale un `select_slider` pour le seuil (de $10^{-6}$ à $10^{-2}$, défaut $10^{-3}$) et affichez le résultat.

<details>
<summary><b>Indice</b></summary>

La moyenne (`np.mean`) d'un tableau de booléens donne la proportion de `True`.

</details>


**Question :** pour $\ln(1+x)$, que vaut la comparaison avec `nan` ? Quel effet sur le taux quand l'intervalle contient des $x \le -1$ ?

### 5.2. Courbe d'erreur

**Consigne :** ajoutez une case à cocher "Afficher l'erreur". Si elle est cochée, affichez $|f(x) - P_n(x)|$ en échelle logarithmique en ordonnée.

<details>
<summary><b>Indice</b></summary>

Une case à cocher renvoie un booléen, utilisable dans un `if`. Un axe Matplotlib a une méthode pour passer en échelle log.

</details>

---

## 6. Comparer plusieurs ordres

Comme vu en 1.3, le script est réexécuté à chaque interaction : on ne se "souvient" de rien. On veut conserver plusieurs ordres pour les comparer sur un même graphique.

**Consigne :** ajoutez un bouton "Ajouter cet ordre à la comparaison". Chaque clic ajoute `n` à une liste conservée entre les interactions (sans doublon). Tracez ensuite $f$ et tous les $P_k$ des ordres sélectionnés. Ajoutez un bouton de réinitialisation.

<details>
<summary><b>Indice 1</b></summary>

Renseignez-vous sur `st.session_state` : il se comporte comme un dictionnaire qui persiste entre deux exécutions.

</details>

<details>
<summary><b>Indice 2</b></summary>

Initialisez la clé une seule fois, en vérifiant qu'elle n'existe pas déjà :

```python
if "ordres" not in st.session_state:
    st.session_state.ordres = []
```

</details>

---

## Vérification finale

- `streamlit run app.py` s'exécute sans erreur.
- Les cinq fonctions s'affichent correctement pour différents ordres.
- Pour $n = 0$, le polynôme est une constante.
- Pour $\sin$ au degré 10 sur $[-2\pi, 2\pi]$, le taux de coïncidence à $10^{-3}$ correspond à celui trouvé en 3.2.
- L'erreur diminue quand $n$ augmente, pour $x$ proche de $0$.
- Le bouton de comparaison conserve les courbes entre deux interactions.

---

## Pour aller plus loin

1. Remplacez le catalogue par un champ libre (`st.text_input`) saisissant une expression en `x`, évaluée avec `eval` dans un environnement restreint contenant `np` ; gérez l'erreur avec `try/except` et `st.error`.
2. Affichez sous forme de tableau (`st.dataframe`) les coefficients $f^{(k)}(0)/k!$ pour $k$ de $0$ à $n$.
3. Estimez, à partir de la courbe d'erreur, l'intervalle autour de $0$ sur lequel l'erreur reste inférieure au seuil choisi.
4. Généralisez à un point de développement $a$ quelconque : quelles lignes de `taylor1` et `taylor2` changent ?
