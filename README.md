# 🚗 Projet RL : Voiture Autonome

Ce projet implémente un agent de conduite autonome utilisant l'apprentissage par renforcement (Reinforcement Learning) dans un environnement de trafic dense simulé.

**Auteurs :** Amine, Zakaria, Wajih, Thomas.

## 🛠 Installation

Avant de lancer le projet, assurez-vous d'avoir Python installé et installez les librairies nécessaires :

```bash
pip install gymnasium highway-env stable-baselines3 numpy shimmy

```

## 🚀 Comment lancer le projet

Le projet est divisé en deux versions principales.

### 1. Version V0 : Contrôle Longitudinal (Vitesse)

Cette version gère uniquement l'accélération et le freinage sur une seule voie.

* **Pour lancer l'entraînement :**
```bash
python version_v0.py

```


* **Pour tester le modèle (Visualisation) :**
```bash
python version_v0.py test

```



### 2. Version V1 : Contrôle Latéral (Changement de voie)

Cette version apprend à la voiture à doubler et changer de voie pour optimiser son temps de trajet.

* **Étape 1 : Entraîner le modèle**
Lancez ce script pour que l'IA apprenne (création du fichier de sauvegarde) :
```bash
python train_v1.py

```


* **Étape 2 : Voir le résultat (Démo)**
Une fois l'entraînement fini, lancez ce script pour voir la voiture conduire toute seule :
```bash
python evaluate_v1.py

```



## 📂 Structure des fichiers

* `version_v0.py` : Code complet pour la V0 (Entraînement et Test via PPO).
* `train_v1.py` : Script d'entraînement pour la V1 (DQN).
* `evaluate_v1.py` : Script de démonstration pour la V1 (charge le modèle entraîné).
* `models/` : Dossier où sont sauvegardés les modèles entraînés.
        train()

```
