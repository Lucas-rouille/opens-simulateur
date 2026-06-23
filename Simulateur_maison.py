"""

Simulateur de courbe de charge résidentielle  (v1)
===================================================
Interface Streamlit — 4 onglets :
  🏠 Foyer        → composition, tranches d'âge, horaires d'occupation
  ⚙️ Équipements  → activer/désactiver, probabilités, forçage manuel
  📅 Simulation   → nombre de jours, lancer
  📊 Résultats    → courbes moyennes semaine/weekend/vacances + stackplot
Lancer :
    streamlit run simulateur_maison.py
"""

import datetime
import random
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
#  CHEMINS
# ═══════════════════════════════════════════════════════════════════════════════
# Dossier unique des cycles arbitrés (généré par arbitre.py)
# Format : {usage}_{source}_cycle{N}.csv  (source = LPG / SmartHouse / REFIT)
CYCLES_FINAUX_DIR = Path("DATA/Cycles/cycles_finaux_simulateur")
METEO_CSV = None
METEO_STATION = "RENNES-ST JACQUES"

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
MINUTES_DAY   = 1440
SEUIL_STACKPLOT_W = 2    # En dessous → catégorie "Autre"
# Tranches d'âge
AGE_GROUPS = ["Bébé (0-3 ans)", "Enfant (4-11 ans)",
              "Adolescent (12-17 ans)", "Adulte actif (18-65 ans)",
              "Senior (65+ ans)"]

# ═══════════════════════════════════════════════════════════════════════════════
#  PROBABILITÉS D'USAGE PAR PROFIL INSEE (source : Enquête Emploi du Temps INSEE)
#  Fréquence journalière observée = probabilité qu'un individu du profil
#  pratique cette activité un jour de semaine donné.
#  Pondération automatique selon composition du foyer.

# ═══════════════════════════════════════════════════════════════════════════════
PROB_INSEE = {
    "Adulte actif (18-65 ans)": {
        "hob":             0.59,
        "dishwasher":      0.33,
        "washing_machine": 0.17,
        "vacuum":          0.47,
        "tv":              0.70,
        "garden":          0.08,
        "oven":            0.59,   # inclus dans cuisine
    },
    "Adolescent (12-17 ans)": {
        "hob":             0.56,
        "dishwasher":      0.29,
        "washing_machine": 0.04,
        "vacuum":          0.42,
        "tv":              0.57,
        "garden":          0.00,
        "oven":            0.56,
    },
    "Senior (65+ ans)": {   # identique retraité
        "hob":             0.71,
        "dishwasher":      0.50,
        "washing_machine": 0.20,
        "vacuum":          0.65,
        "tv":              0.87,
        "garden":          0.29,
        "oven":            0.71,
    },
}
# Profils sans données → adulte actif par défaut
for _age in ["Bébé (0-3 ans)", "Enfant (4-11 ans)"]:
    PROB_INSEE[_age] = PROB_INSEE["Adulte actif (18-65 ans)"]
# Mapping tranche d'âge → profil EEDT par défaut
# Chaque tranche a son profil de distribution horaire et de présence associé
AGE_TO_EEDT_PROFIL = {
    "Bébé (0-3 ans)":          None,            # pas de profil EDT → Personnalisé
    "Enfant (4-11 ans)":        None,            # pas de profil EDT → Personnalisé
    "Adolescent (12-17 ans)":   None,            # pas de profil EDT → Personnalisé
    "Adulte actif (18-65 ans)": "Adulte actif",
    "Senior (65+ ans)":         "Retraité",
}

def compute_weighted_probs(members):

    """

    Calcule les probabilités pondérées selon la composition du foyer.
    Retourne {usage: prob_pondérée} pour les usages avec données INSEE.
    """

    if not members:
        return {}
    usages = list(next(iter(PROB_INSEE.values())).keys())
    weighted = {}
    for usage in usages:
        probs = [PROB_INSEE.get(
            m.get("age_group", "Adulte actif (18-65 ans)"), {}
        ).get(usage, 0.0) for m in members]
        weighted[usage] = round(float(np.mean(probs)), 3)
    return weighted

# ═══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS HORAIRES INSEE (source : Enquête Emploi du Temps INSEE)
#  Vecteur 24h de poids proportionnels à la probabilité d'usage à cette heure.
#  Extraits des courbes de distribution horaire des carnets d'activité.

# ═══════════════════════════════════════════════════════════════════════════════

def _w(pairs):

    v = np.zeros(24)
    for h, w in pairs:
        v[int(h)] = w
    s = v.sum()
    return (v / s).tolist() if s > 0 else (np.ones(24)/24).tolist()
HOURLY_WEIGHTS_INSEE = {
    "Adulte actif": {
        "hob":             _w([(12,2),(13,1),(19,4),(20,3)]),
        "oven":            _w([(12,2),(13,1),(19,4),(20,3)]),
        "dishwasher":      _w([(13,2),(20,4),(21,3),(22,1)]),
        "washing_machine": _w([(h,1) for h in range(7, 21)]),
        "vacuum":          _w([(h,1) for h in range(7, 21)]),
        "tv":              _w([(18,1),(19,2),(20,4),(21,4),(22,3)]),
        "garden":          _w([(h,1) for h in range(7, 20)]),
        "coffee_machine":  _w([(7,3),(8,2),(12,1),(16,1)]),
        "microwave":       _w([(12,3),(13,1),(19,3),(20,2)]),
        "hair_dryer":      _w([(h,1) for h in range(7, 22)]),
        "desktop_pc":      _w([(h,1) for h in range(7, 23)]),
        "lights":          _w([(6,1),(7,2),(8,1),(17,2),(18,3),(19,3),(20,3),(21,2),(22,1)]),
    },
    "Retraité": {
        "hob":             _w([(11,2),(12,4),(13,2),(18,2),(19,3),(20,2)]),
        "oven":            _w([(11,2),(12,4),(13,2),(18,2),(19,3),(20,2)]),
        "dishwasher":      _w([(12,3),(13,3),(19,2),(20,2),(21,1)]),
        "washing_machine": _w([(9,2),(10,3),(11,3),(14,2),(15,1)]),
        "vacuum":          _w([(9,3),(10,4),(11,3),(14,2)]),
        "tv":              _w([(14,2),(15,2),(20,4),(21,4),(22,2)]),
        "garden":          _w([(9,2),(10,3),(11,2),(15,2),(16,2)]),
        "coffee_machine":  _w([(7,2),(8,3),(10,2),(15,2),(16,1)]),
        "microwave":       _w([(12,3),(13,2),(19,2),(20,2)]),
        "hair_dryer":      _w([(7,2),(8,3),(9,2)]),
        "desktop_pc":      _w([(h,1) for h in range(9, 22)]),
        "lights":          _w([(7,1),(8,1),(17,2),(18,3),(19,3),(20,3),(21,2),(22,1)]),
    },
    "Étudiant": {
        "hob":             _w([(12,2),(13,2),(19,3),(20,3)]),
        "oven":            _w([(12,2),(13,2),(19,3),(20,3)]),
        "dishwasher":      _w([(13,2),(20,3),(21,3),(22,1)]),
        "washing_machine": _w([(h,1) for h in range(9, 22)]),
        "vacuum":          _w([(h,1) for h in range(9, 22)]),
        "tv":              _w([(17,1),(18,2),(20,3),(21,4),(22,3)]),
        "garden":          _w([]),
        "coffee_machine":  _w([(8,2),(9,2),(15,2),(16,1)]),
        "microwave":       _w([(12,3),(13,2),(19,3),(20,2)]),
        "hair_dryer":      _w([(h,1) for h in range(7, 23)]),
        "desktop_pc":      _w([(h,1) for h in range(9, 24)]),
        "lights":          _w([(18,2),(19,3),(20,3),(21,3),(22,2)]),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILS DE PRÉSENCE INSEE (source : Enquête Emploi du Temps INSEE)
#  Probabilité d'être au domicile par heure pour chaque profil.

# ═══════════════════════════════════════════════════════════════════════════════
PRESENCE_PROFILES = {
    "Personnalisé":              None,   # créneaux manuels
    "Adulte actif": [0.97,0.98,0.97,0.97,0.97,0.93,0.89,0.65,
                                   0.33,0.22,0.20,0.27,0.33,0.26,0.23,0.25,
                                   0.29,0.43,0.54,0.68,0.81,0.86,0.91,0.94],
    "Retraité": [0.998,1.0,1.0,1.0,1.0,0.997,0.993,0.97,
                                   0.93,0.88,0.83,0.87,0.88,0.84,0.83,0.80,
                                   0.82,0.86,0.87,0.91,0.94,0.96,0.978,0.992],
    "Étudiant": [0.91,0.99,0.99,0.99,0.99,0.99,0.985,0.82,
                                   0.53,0.36,0.32,0.36,0.39,0.35,0.31,0.30,
                                   0.34,0.42,0.47,0.57,0.66,0.72,0.77,0.86],
}
# Périodes de vacances par défaut (mois, semaine_du_mois)
# Format : liste de (date_debut, date_fin) en mm-jj
VACANCES_DEFAULT = [
    ("07-05", "07-19"),   # 2 semaines été
    ("10-26", "11-01"),   # Toussaint
    ("12-21", "12-27"),   # Noël
    ("04-12", "04-18"),   # Pâques
]
# Couleurs par usage pour le stackplot
USAGE_COLORS = {
    # Froid
    "fridge":           "#1565C0",   # bleu foncé
    # Cuisson
    "hob":              "#FF6D00",   # orange vif
    "oven":             "#E64A19",   # rouge-orange
    "microwave":        "#FFB300",   # ambre
    "bread_maker":      "#F9A825",   # jaune doré
    "extractor_hood":   "#FFD54F",   # jaune clair
    "small_cooking":    "#FFCC02",   # jaune
    "coffee_machine":   "#6D4C41",   # marron
    # Électroménager
    "washing_machine":  "#7B1FA2",   # violet foncé
    "dishwasher":       "#AB47BC",   # violet moyen
    "dryer":            "#CE93D8",   # violet clair
    "vacuum":           "#4CAF50",   # vert
    "hair_dryer":       "#E91E63",   # rose
    # Divertissement
    "tv":               "#00ACC1",   # cyan foncé
    "sat_hifi":         "#26C6DA",   # cyan clair
    "gaming":           "#00838F",   # cyan très foncé
    "beamer":           "#80DEEA",   # cyan très clair
    # Informatique
    "desktop_pc":       "#283593",   # indigo foncé
    "printer":          "#5C6BC0",   # indigo clair
    # Éclairage
    "lights":           "#FFF176",   # jaune très clair
    # Climatisation / jardin
    "air_conditioner":  "#26A69A",   # teal
    "garden":           "#388E3C",   # vert foncé
    # Fond
    "Bruit_fond":       "#EEEEEE",   # gris très clair
    "Autre":            "#90A4AE",   # gris bleuté
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CATALOGUE DES APPAREILS
# Clé : nom_csv (sans extension)
# Valeur :
#   label         → nom affiché
#   prob_semaine  → probabilité d'occurrence par jour de semaine (0-1)
#   prob_weekend  → probabilité par jour de weekend
#   prob_vacances → probabilité par jour de vacances
#   heure_pic     → heure de démarrage typique (0-23)
#   sigma_h       → écart-type en heures
#   besoin_presence→ True = ne tourne que si ≥1 personne présente
#   bruit_fond    → True = tourne en arrière-plan même si absent
#                   si bruit_fond, on approxime par bruit blanc (puissance_W)
#   puissance_fond→ puissance du bruit blanc en W (si bruit_fond=True)
#   age_requis    → liste de tranches d'âge qui peuvent l'utiliser ([] = tous)
#   semaine_h_pic → heure pic semaine (None = même que heure_pic)
#   weekend_h_pic → heure pic weekend

# ═══════════════════════════════════════════════════════════════════════════════
CATALOGUE = {
    "freezer": {
        "label": "Congélateur", "prob_semaine": 1.0, "prob_weekend": 1.0,
        "prob_vacances": 1.0, "heure_pic": None, "sigma_h": None,
        "besoin_presence": False, "bruit_fond": False, "puissance_fond": 0,
        "continu": True,
        "gap_off_min": 60, "gap_off_max": 90,
        "age_requis": [],
        "semaine_h_pic": None, "weekend_h_pic": None,
    },
    "fridge": {
        "label": "Réfrigérateur", "prob_semaine": 1.0, "prob_weekend": 1.0,
        "prob_vacances": 1.0, "heure_pic": None, "sigma_h": None,
        "besoin_presence": False, "bruit_fond": False, "puissance_fond": 0,
        "continu": True,   # cycles enchaînés en continu toute la journée
        "gap_off_min": 60, "gap_off_max": 90,  # gap OFF entre cycles (min)
        "age_requis": [],
        "semaine_h_pic": None, "weekend_h_pic": None,
    },
    "washing_machine": {
        "label": "Lave-linge", "prob_semaine": 0.3, "prob_weekend": 0.5,
        "prob_vacances": 0.0, "heure_pic": 10, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 10, "weekend_h_pic": 11,
    },
    "dishwasher": {
        "label": "Lave-vaisselle", "prob_semaine": 0.6, "prob_weekend": 0.7,
        "prob_vacances": 0.0, "heure_pic": 20, "sigma_h": 1.5,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 20, "weekend_h_pic": 19,
    },
    "oven": {
        "label": "Four", "prob_semaine": 0.3, "prob_weekend": 0.5,
        "prob_vacances": 0.0, "heure_pic": 18, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 18, "weekend_h_pic": 12,
    },
    "hob": {
        "label": "Plaques de cuisson", "prob_semaine": 0.7, "prob_weekend": 0.8,
        "prob_vacances": 0.0, "heure_pic": 19, "sigma_h": 1.5,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 19, "weekend_h_pic": 12,
    },
    "microwave": {
        "label": "Micro-ondes", "prob_semaine": 0.6, "prob_weekend": 0.5,
        "prob_vacances": 0.0, "heure_pic": 12, "sigma_h": 3,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": [],
        "semaine_h_pic": 12, "weekend_h_pic": 13,
    },
    "bread_maker": {
        "label": "Machine à pain", "prob_semaine": 0.1, "prob_weekend": 0.3,
        "prob_vacances": 0.0, "heure_pic": 7, "sigma_h": 1,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 7, "weekend_h_pic": 9,
    },
    "coffee_machine": {
        "label": "Cafetière", "prob_semaine": 0.8, "prob_weekend": 0.7,
        "prob_vacances": 0.0, "heure_pic": 7, "sigma_h": 1,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 7, "weekend_h_pic": 9,
    },
    "tv": {
        "label": "Télévision", "prob_semaine": 0.8, "prob_weekend": 0.9,
        "prob_vacances": 0.0, "heure_pic": 20, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": True, "puissance_fond": 5,
        "age_requis": [],
        "semaine_h_pic": 20, "weekend_h_pic": 15,
    },
    "sat_hifi": {
        "label": "SAT/HiFi", "prob_semaine": 0.6, "prob_weekend": 0.7,
        "prob_vacances": 0.0, "heure_pic": 20, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": True, "puissance_fond": 8,
        "age_requis": [],
        "semaine_h_pic": 20, "weekend_h_pic": 15,
    },
    "gaming": {
        "label": "Console de jeux", "prob_semaine": 0.3, "prob_weekend": 0.6,
        "prob_vacances": 0.0, "heure_pic": 17, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Enfant (4-11 ans)", "Adolescent (12-17 ans)"],
        "semaine_h_pic": 17, "weekend_h_pic": 14,
    },
    "desktop_pc": {
        "label": "PC fixe", "prob_semaine": 0.5, "prob_weekend": 0.4,
        "prob_vacances": 0.0, "heure_pic": 20, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": True, "puissance_fond": 10,
        "age_requis": ["Adolescent (12-17 ans)", "Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 20, "weekend_h_pic": 14,
    },
    "printer": {
        "label": "Imprimante/Scanner", "prob_semaine": 0.1, "prob_weekend": 0.1,
        "prob_vacances": 0.0, "heure_pic": 10, "sigma_h": 3,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 10, "weekend_h_pic": 10,
    },
    "lights": {
        "label": "Éclairage", "prob_semaine": 1.0, "prob_weekend": 1.0,
        "prob_vacances": 0.0, "heure_pic": 19, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": [],
        "semaine_h_pic": 19, "weekend_h_pic": 18,
    },
    "hair_dryer": {
        "label": "Sèche-cheveux", "prob_semaine": 0.5, "prob_weekend": 0.4,
        "prob_vacances": 0.0, "heure_pic": 8, "sigma_h": 1,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adolescent (12-17 ans)", "Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 8, "weekend_h_pic": 9,
    },
    "vacuum": {
        "label": "Aspirateur", "prob_semaine": 0.2, "prob_weekend": 0.4,
        "prob_vacances": 0.0, "heure_pic": 10, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 10, "weekend_h_pic": 10,
    },
    "extractor_hood": {
        "label": "Hotte aspirante", "prob_semaine": 0.5, "prob_weekend": 0.6,
        "prob_vacances": 0.0, "heure_pic": 19, "sigma_h": 1.5,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 19, "weekend_h_pic": 12,
    },
    "small_cooking": {
        "label": "Petit électroménager cuisine", "prob_semaine": 0.2,
        "prob_weekend": 0.4, "prob_vacances": 0.0, "heure_pic": 18, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 18, "weekend_h_pic": 12,
    },
    "beamer": {
        "label": "Vidéoprojecteur", "prob_semaine": 0.05, "prob_weekend": 0.15,
        "prob_vacances": 0.0, "heure_pic": 20, "sigma_h": 1,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 20, "weekend_h_pic": 20,
    },
    "dryer": {
        "label": "Sèche-linge", "prob_semaine": 0.15, "prob_weekend": 0.3,
        "prob_vacances": 0.0, "heure_pic": 11, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 11, "weekend_h_pic": 11,
    },
    "air_conditioner": {
        "label": "Climatiseur", "prob_semaine": 0.2, "prob_weekend": 0.3,
        "prob_vacances": 0.0, "heure_pic": 14, "sigma_h": 3,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": [],
        "semaine_h_pic": 14, "weekend_h_pic": 14,
    },
    "garden": {
        "label": "Outils de jardin", "prob_semaine": 0.03, "prob_weekend": 0.12,
        "prob_vacances": 0.0, "heure_pic": 10, "sigma_h": 2,
        "besoin_presence": True, "bruit_fond": False, "puissance_fond": 0,
        "age_requis": ["Adulte actif (18-65 ans)", "Senior (65+ ans)"],
        "semaine_h_pic": 10, "weekend_h_pic": 10,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS HORAIRES PAR USAGE
#  Vecteur de 24 poids (index = heure), normalisé automatiquement.
#  Remplace la gaussienne par un tirage pondéré sur les heures disponibles.

# ═══════════════════════════════════════════════════════════════════════════════

def _w(pairs):

    """Construit un vecteur 24h depuis une liste (heure, poids)."""

    v = np.zeros(24)
    for h, w in pairs:
        v[int(h)] = w
    s = v.sum()
    return (v / s).tolist() if s > 0 else (np.ones(24)/24).tolist()
HOURLY_WEIGHTS_SEMAINE = {
    # Cuisine — concentrée aux repas
    "oven":            _w([(12,2),(13,2),(19,3),(20,3)]),
    "hob":             _w([(12,2),(13,2),(19,3),(20,3)]),
    "microwave":       _w([(12,2),(13,1),(19,2),(20,1)]),
    "coffee_machine":  _w([(7,3),(8,2),(12,1),(15,1),(16,1)]),
    "small_cooking":   _w([(7,1),(8,2),(12,2),(19,2),(20,1)]),
    "bread_maker":     _w([(6,2),(7,3),(8,1)]),
    "dishwasher":      _w([(20,3),(21,3),(22,1)]),
    # Électroménager — large plage 7h-21h équiprobable
    "washing_machine": _w([(h,1) for h in range(7, 21)]),
    "vacuum":          _w([(h,1) for h in range(7, 21)]),
    "hair_dryer":      _w([(h,1) for h in range(7, 22)]),
    "garden":          _w([(h,1) for h in range(7, 20)]),
    # Divertissement — soir
    "tv":              _w([(18,1),(19,2),(20,3),(21,3),(22,2)]),
    "gaming":          _w([(16,1),(17,2),(18,2),(19,2),(20,2),(21,1)]),
    "sat_hifi":        _w([(h,1) for h in range(7, 23)]),
    "beamer":          _w([(20,2),(21,3),(22,2)]),
    # Informatique — large plage 7h-22h
    "desktop_pc":      _w([(h,1) for h in range(7, 23)]),
    "printer":         _w([(h,1) for h in range(7, 23)]),
    # Éclairage — matin + soirée
    "lights":          _w([(6,1),(7,2),(8,1),(17,2),(18,3),(19,3),(20,3),(21,2),(22,1)]),
    # Climatisation — après-midi chaud
    "air_conditioner": _w([(13,2),(14,3),(15,3),(16,2),(17,1)]),
    "electric_heater": _w([(6,2),(7,3),(8,1),(18,2),(19,2)]),
}
# Weekend : mélange distribution semaine × uniforme (α=0.5)
WEEKEND_ALPHA = 0.5   # 0=totalement uniforme, 1=identique à semaine

# ═══════════════════════════════════════════════════════════════════════════════
#  ÉCLAIRAGE — modélisation continue basée sur présence + ensoleillement

# ═══════════════════════════════════════════════════════════════════════════════
# Heure de lever/coucher du soleil par saison (heure locale France)
SOLEIL = {
    "hiver":     {"lever": 8.5,  "coucher": 16.5},
    "printemps": {"lever": 6.5,  "coucher": 20.5},
    "été":       {"lever": 5.5,  "coucher": 21.5},
    "automne":   {"lever": 7.5,  "coucher": 18.5},
}
# Puissance d'éclairage de base par membre présent (W)
LIGHTS_W_PER_MEMBER = 25.0
# Facteur d'éclairage selon l'heure (0=plein soleil, 1=nuit noire)

def daylight_factor(minute: int, season: str) -> float:

    """Retourne 0 si plein jour, 1 si nuit noire, interpolation crépuscule ±1h."""

    h = minute / 60.0
    lever   = SOLEIL[season]["lever"]
    coucher = SOLEIL[season]["coucher"]
    # Nuit complète
    if h < lever - 1 or h > coucher + 1:
        return 1.0
    # Plein jour
    if lever + 1 <= h <= coucher - 1:
        return 0.0
    # Crépuscule matin
    if h < lever + 1:
        return max(0.0, 1.0 - (h - (lever - 1)) / 2.0)
    # Crépuscule soir
    return max(0.0, (h - (coucher - 1)) / 2.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAUX D'IMMOBILISATION PAR USAGE
#  0.0 = usage passif (lave-linge, four) → membre libre pour autre chose
#  1.0 = usage actif exclusif (sèche-cheveux) → membre bloqué pendant le cycle

# ═══════════════════════════════════════════════════════════════════════════════
IMMOBILISATION_DEFAULT = {
    "hair_dryer":     1.0,
    "gaming":         1.0,
    "vacuum":         0.9,
    "tv":             0.8,
    "beamer":         0.8,
    "desktop_pc":     0.7,
    "hob":            0.5,
    "sat_hifi":       0.3,
    "microwave":      0.1,
    "coffee_machine": 0.1,
    "lights":         0.0,
    "washing_machine":0.0,
    "dishwasher":     0.0,
    "oven":           0.0,
    "dryer":          0.0,
    "extractor_hood": 0.0,
    "bread_maker":    0.0,
    "printer":        0.2,
    "small_cooking":  0.2,
    "garden":         0.8,
    "air_conditioner":0.0,
    "electric_heater":0.0,
}

def _blend_weekend(usage):

    sem = HOURLY_WEIGHTS_SEMAINE.get(usage, None)
    if sem is None:
        return None
    uniform = [1/24] * 24
    return [WEEKEND_ALPHA * s + (1 - WEEKEND_ALPHA) * u
            for s, u in zip(sem, uniform)]
HOURLY_WEIGHTS_WEEKEND = {u: _blend_weekend(u)
                          for u in HOURLY_WEIGHTS_SEMAINE}

def draw_start_density(weights_24h: list, presence_mask: np.ndarray,
                       cycle_duration: int, max_tries: int = 100) -> int | None:
    w = np.array(weights_24h, dtype=float)
    w = w / w.sum()
    hours = np.arange(24)
    for _ in range(max_tries):
        h = int(np.random.choice(hours, p=w))
        m = random.randint(0, 59)
        t = h * 60 + m
        # Évite un pic artificiel à 23h59 si un cycle est tronqué net.
        t = max(0, min(MINUTES_DAY - 1, t))
        if presence_mask[t]:
            return t
    return None

# ═══════════════════════════════════════════════════════════════════════════════
#  FACTEURS SAISONNIERS

# ═══════════════════════════════════════════════════════════════════════════════
# Facteur multiplicatif sur prob_semaine/prob_weekend selon la saison.
# Saisons : hiver=déc-fév, printemps=mar-mai, été=jun-août, automne=sep-nov
SEASONAL_FACTORS = {
    "lights": {
        "hiver": 1.4, "printemps": 0.9, "été": 0.7, "automne": 1.1,
        # Heure de pic variable : éclairage plus tôt en hiver
        "h_pic_hiver": 17.0, "h_pic_printemps": 19.5,
        "h_pic_été": 21.0,   "h_pic_automne": 18.5,
    },
    "air_conditioner": {
        "hiver": 0.0, "printemps": 0.2, "été": 1.5, "automne": 0.1,
    },
    "oven": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.8, "automne": 1.1,
    },
    "hob": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.8, "automne": 1.1,
    },
    "small_cooking": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.8, "automne": 1.1,
    },
    "washing_machine": {
        "hiver": 0.9, "printemps": 1.1, "été": 1.2, "automne": 1.0,
    },
    "dryer": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.7, "automne": 1.1,
    },
    "tv": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.8, "automne": 1.1,
    },
    "sat_hifi": {
        "hiver": 1.2, "printemps": 1.0, "été": 0.8, "automne": 1.1,
    },
    "garden": {
        "hiver": 0.0, "printemps": 1.2, "été": 1.5, "automne": 0.5,
    },
}
# Facteur par défaut pour tous les usages non listés
SEASONAL_DEFAULT = {"hiver": 1.0, "printemps": 1.0, "été": 1.0, "automne": 1.0}

# ═══════════════════════════════════════════════════════════════════════════════
#  FACTEURS MÉTÉO (remplacent/affinent les facteurs saisonniers si dispo)

# ═══════════════════════════════════════════════════════════════════════════════
# Seuils de température (°C) — calibrés pour le climat océanique tempéré (Rennes)
T_CHAUD      = 20.0   # au-dessus → climatiseur probable, four/TV réduits
T_TRES_CHAUD = 25.0   # journée chaude (rare à Rennes)
T_FRAIS      = 12.0   # en dessous → chauffage sdb, éclairage accru
T_FROID      =  4.0   # en dessous → très froid, éclairage max
# Facteur multiplicatif sur prob selon température moyenne du jour
# Interpolation linéaire entre les seuils
METEO_FACTORS = {
    "lights": {
        # Plus il fait chaud = plus il fait jour = moins d'éclairage
        T_TRES_CHAUD: 0.65,
        T_CHAUD:      0.75,
        T_FRAIS:      1.1,
        T_FROID:      1.4,
    },
    "air_conditioner": {
        T_TRES_CHAUD: 2.0,
        T_CHAUD:      1.0,
        T_FRAIS:      0.05,
        T_FROID:      0.0,
    },
    "oven": {
        T_TRES_CHAUD: 0.6,
        T_CHAUD:      0.8,
        T_FRAIS:      1.1,
        T_FROID:      1.3,
    },
    "hob": {
        T_TRES_CHAUD: 0.7,
        T_CHAUD:      0.85,
        T_FRAIS:      1.1,
        T_FROID:      1.25,
    },
    "small_cooking": {
        T_TRES_CHAUD: 0.7,
        T_CHAUD:      0.85,
        T_FRAIS:      1.1,
        T_FROID:      1.25,
    },
    "tv": {
        T_TRES_CHAUD: 0.75,
        T_CHAUD:      0.85,
        T_FRAIS:      1.1,
        T_FROID:      1.25,
    },
    "sat_hifi": {
        T_TRES_CHAUD: 0.75,
        T_CHAUD:      0.85,
        T_FRAIS:      1.1,
        T_FROID:      1.25,
    },
    "garden": {
        T_TRES_CHAUD: 0.6,   # trop chaud → on ne jardine pas
        T_CHAUD:      1.2,
        T_FRAIS:      0.8,
        T_FROID:      0.0,
    },
    "washing_machine": {
        T_TRES_CHAUD: 1.3,   # on transpire plus
        T_CHAUD:      1.2,
        T_FRAIS:      1.0,
        T_FROID:      0.9,
    },
}

def meteo_factor(usage: str, temp_c: float) -> float:

    """

    Retourne un facteur multiplicatif sur la probabilité d'un usage
    en fonction de la température moyenne du jour.
    Interpolation linéaire entre les seuils définis.
    Si usage non défini → retourne 1.0.
    """

    if usage not in METEO_FACTORS or temp_c is None or np.isnan(temp_c):
        return 1.0
    thresholds = METEO_FACTORS[usage]
    temps_sorted = sorted(thresholds.keys())   # croissant
    if temp_c <= temps_sorted[0]:
        return thresholds[temps_sorted[0]]
    if temp_c >= temps_sorted[-1]:
        return thresholds[temps_sorted[-1]]
    # Interpolation linéaire entre les deux seuils encadrants
    for i in range(len(temps_sorted) - 1):
        t_lo = temps_sorted[i]
        t_hi = temps_sorted[i + 1]
        if t_lo <= temp_c <= t_hi:
            f_lo = thresholds[t_lo]
            f_hi = thresholds[t_hi]
            alpha = (temp_c - t_lo) / (t_hi - t_lo)
            return round(f_lo + alpha * (f_hi - f_lo), 3)
    return 1.0

def get_season(date: datetime.date) -> str:

    m = date.month
    if m in (12, 1, 2):  return "hiver"
    if m in (3, 4, 5):   return "printemps"
    if m in (6, 7, 8):   return "été"
    return "automne"

def seasonal_factor(usage: str, season: str) -> float:

    return SEASONAL_FACTORS.get(usage, SEASONAL_DEFAULT).get(season, 1.0)

def seasonal_h_pic(usage: str, season: str, base_h_pic: float) -> float:

    """Retourne l'heure de pic ajustée selon la saison (uniquement pour lights)."""

    key = f"h_pic_{season}"
    return SEASONAL_FACTORS.get(usage, {}).get(key, base_h_pic)

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION USAGES PERSONNELS vs FOYER

# ═══════════════════════════════════════════════════════════════════════════════
# Usages "personnels" : peuvent générer N événements/jour (un par membre)
PERSONAL_USAGES = {
    "tv", "gaming", "desktop_pc", "sat_hifi", "beamer",
    "hair_dryer", "printer",
}
# Tous les autres usages sont "foyer" (1 seul événement/jour)

def compute_equiv_persons(members, usage, age_factors_ov):

    """

    Calcule n_equiv pour la loi binomiale :
    somme des facteurs âge de chaque membre pour cet usage.
    Remplace le nombre brut de membres par une somme pondérée.
    """

    af_ov = age_factors_ov or {}
    total = 0.0
    for m in members:
        age = m.get("age_group", "Adulte actif (18-65 ans)")
        if age in af_ov:
            af = float(af_ov[age].get(usage, 1.0))
        else:
            af = float(AGE_FACTORS.get(age, {}).get(usage, 1.0))
        total += af
    return max(total, 0.01)
# Facteur d'usage par tranche d'âge (multiplie la probabilité)
AGE_FACTORS = {
    "Bébé (0-3 ans)":          {"lights": 0.3, "tv": 0.1, "gaming": 0.0,
                                  "desktop_pc": 0.0},
    "Enfant (4-11 ans)":        {"gaming": 1.5, "tv": 1.2},
    "Adolescent (12-17 ans)":   {"gaming": 2.0, "tv": 1.2, "desktop_pc": 1.5},
    "Adulte actif (18-65 ans)": {},
    "Senior (65+ ans)":         {"tv": 1.5, "gaming": 0.0,
                                  "desktop_pc": 0.3, "lights": 1.2},
}
# Facteur d'échelle éclairage par nombre de pièces (simplifié par nb personnes)

def lights_factor(n_persons: int) -> float:

    return max(1.0, 0.5 + 0.4 * n_persons)

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES CYCLES

# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data

def load_meteo(meteo_csv: Path, station: str, annee: int) -> dict:

    """

    Charge les données météo pour une station et une année données.
    Retourne un dict {date: temp_moy_C} avec une entrée par jour calendaire.
    """

    if meteo_csv.empty:
        return {}
    try:
        # On utilise directement le DataFrame au lieu de le relire !
        df = meteo_csv.copy()
        
        df = df[df["NOM_USUEL"] == station].copy()
        df["date"] = pd.to_datetime(df["AAAAMMJJHH"].astype(str),
                                    format="%Y%m%d%H", errors="coerce")
        df["T"] = pd.to_numeric(df["T"], errors="coerce")
        df = df.dropna(subset=["date", "T"])
        # Filtre température aberrante (>45°C ou <-20°C)
        df = df[(df["T"] > -20) & (df["T"] < 45)]
        df_year = df[df["date"].dt.year == annee].copy()
        df_year["day"] = df_year["date"].dt.date
        # Température moyenne journalière
        daily = df_year.groupby("day")["T"].mean().round(2)
        return {k: float(v) for k, v in daily.items()}
    except Exception as e:
        return {}
@st.cache_data

def get_available_years(meteo_csv: Path, station: str) -> list:

    """Retourne les années disponibles pour la station."""

    if meteo_csv.empty:
        return [2023]
    try:
        # On utilise directement le DataFrame
        df = meteo_csv.copy()
        
        df = df[df["NOM_USUEL"] == station]
        years = sorted(df["AAAAMMJJHH"].astype(str).str[:4]
                       .astype(int).unique().tolist())
        return years
    except Exception:
        return [2023]
# Source des cycles — rempli dynamiquement par load_all_cycles()
CYCLE_SOURCE = {}

# Plage de durée acceptable par usage [min_min, max_min]
# Source : calibration sur les cycles disponibles + cohérence physique
# ═══════════════════════════════════════════════════════════════════════════════
#  LIAISONS LOGIQUES PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════════════════════
LIAISONS_DEFAULT = [
    {
        "source": "washing_machine", "cible": "dryer",
        "delai_min": 5,   "delai_max": 30,
        "prob": 0.8,      "attend_presence": True,
        "active": True,   "note": "Sèche-linge après Lave-linge",
    },
    {
        "source": "hob", "cible": "extractor_hood",
        "delai_min": 0,   "delai_max": 5,
        "prob": 0.9,      "attend_presence": False,
        "active": True,   "note": "Hotte pendant cuisson",
    },
    {
        "source": "hob", "cible": "dishwasher",
        "delai_min": 30,  "delai_max": 120,
        "prob": 0.7,      "attend_presence": False,
        "active": True,   "note": "Lave-vaisselle après repas (70%)",
    },
    {
        "source": "coffee_machine", "cible": "bread_maker",
        "delai_min": -60, "delai_max": -30,
        "prob": 0.4,      "attend_presence": False,
        "active": True,   "note": "Machine a pain avant le cafe (40%)",
    },
]

CYCLE_DURATION_FILTER = {
    "washing_machine": [60, 120],
    "dishwasher":      [60, 130],
    "dryer":           [20, 80],
    "oven":            [15, 60],
    "hob":             [5,  60],
    "vacuum":          [15, 60],
    "hair_dryer":      [10, 40],
    "garden":          [20, 90],
}

# Puissance moyenne max acceptable par cycle (W) — filtre les cycles aberrants
MAX_CYCLE_MEAN_W = {
    "garden":        900,
    "small_cooking": 800,
}

import numpy as np
import math

def apply_cycle_calibration(cycles: list, cat: dict, usage: str = "") -> list:
    """
    Calibrage strict hybride avec entonnoir et garde-fou anti-pic.
    1. Offset bruit : Auto-calculé sur la moyenne annuelle (Frigo).
    2. Calibrage Électronique : Scaling d'amplitude.
    3. Calibrage Thermomécanique : Répétition N, puis Scaling, puis Étirement X si pic > 3500W.
    """
    if not cycles:
        return cycles

    # 1. OFFSET BRUIT (Correction Thermodynamique pour le Frigo)
    noise_w = cat.get("power_noise_floor_w")
    annual_mean_target = cat.get("annual_mean_target_w")
    
    if annual_mean_target and float(annual_mean_target) > 0:
        if usage == "fridge":
            gap_avg = (cat.get("gap_off_min", 60) + cat.get("gap_off_max", 90)) / 2.0
            
            calibrated_fridge = []
            for c in cycles:
                dur_c = len(c)
                if dur_c == 0: continue
                
                # Moyenne lissée sur le temps total (actif + pause)
                mean_sim = c.sum() / (dur_c + gap_avg)
                diff = mean_sim - float(annual_mean_target)
                
                if diff > 0:
                    # LE CORRECTIF PHYSIQUE : On concentre la soustraction sur la phase active
                    ratio_dilution = (dur_c + gap_avg) / dur_c
                    offset_concentre = diff * ratio_dilution
                    print(f"  [Auto-Offset Frigo] Écart de {diff:.1f}W -> Dilution x{ratio_dilution:.1f} -> On enlève {offset_concentre:.1f}W")
                    
                    c_new = np.clip(c - offset_concentre, 0, None)
                    calibrated_fridge.append(c_new)
                else:
                    calibrated_fridge.append(c)
            cycles = calibrated_fridge
    if usage in ("fridge", "freezer"):
        return cycles
    # 2. CALIBRAGE ÉNERGÉTIQUE HYBRIDE (Entonnoir)
    e_target = cat.get("energy_target_wh")
    if e_target and float(e_target) > 0:
        e_target = float(e_target)
        calibrated = []
        
        # Détermination de la durée cible et maximale
        dur_filter = cat.get("dur_filter", [0, 999])
        # Pour les appareils thermomécaniques longs (lave-linge, four…),
        # on autorise jusqu'à 2× la durée max du filtre, plafonné à 720 min.
        dur_max_allowed = min(dur_filter[1] * 2.0, 720)
        
        # ── CORRECTION : Tolérance dynamique pour le Fallback ──
        # Si le cycle qu'on a forcé en entrée est déjà plus long que le plafond, 
        # on élargit le plafond pour autoriser son calibrage.
        max_in_len = max([len(c) for c in cycles]) if cycles else 0
        if max_in_len > dur_max_allowed:
            dur_max_allowed = max_in_len * 1.5
            
        dur_cible = (dur_filter[0] + dur_filter[1]) / 2.0
        
        MAX_POWER_W = 3500.0  # Limite physique absolue (16A)
        SCALING_AMPLITUDE = [ "gaming", "sat_hifi", "beamer", "lights"]

        for c in cycles:
            e_c = c.sum() / 60.0
            dur_c = len(c)
            max_p = c.max() if dur_c > 0 else 0
            if e_c <= 0 or dur_c <= 0 or max_p <= 0: continue
            
            if usage in SCALING_AMPLITUDE:
                # --- MODE ÉLECTRONIQUE (Axe Y) ---
                ratio = e_target / e_c
                if 0.2 <= ratio <= 5.0 and (max_p * ratio) <= MAX_POWER_W:
                    calibrated.append(c * ratio)
                else:
                    calibrated.append(c)
            else:
                # --- MODE THERMOMÉCANIQUE (Entonnoir en 3 étapes) ---
                best_c_new   = None
                best_error   = float('inf')
                
                # Étape 1 : Recherche de la meilleure répétition N
                for N in range(1, 6):
                    e_new   = e_c * N
                    dur_new = dur_c * N
                    
                    if dur_new > dur_max_allowed:
                        continue

                    error = abs(e_new - e_target)
                    if (error <= 0.20 * e_target and abs(dur_new - dur_cible) <= 0.30 * dur_cible):
                        if error < best_error:
                            best_error = error
                            best_c_new = np.tile(c, N)
                            
                # Application de la décision
                if best_c_new is not None:
                    calibrated.append(best_c_new)
                else:
                    # ── CORRECTION : Interdiction absolue de toucher à l'axe Y pour TV/PC ──
                    if usage in ["tv", "desktop_pc"]:
                        # On force la répétition mathématique la plus proche, même si elle 
                        # dépasse les 20% d'erreur, pour garantir que le pic (Y) ne bouge JAMAIS.
                        N_force = max(1, round(e_target / e_c))
                        forced_c = np.tile(c, N_force)
                        
                        if len(forced_c) <= dur_max_allowed:
                            calibrated.append(forced_c)
                        else:
                            calibrated.append(c) # Sécurité : on le laisse brut
                            
                    # ── Fallback standard pour l'électroménager (Four, Lave-Linge...) ──
                    else:
                        # Étape 2 : Fallback via Scaling d'amplitude
                        scale = e_target / e_c
                        new_peak = max_p * scale
                        
                        if new_peak <= MAX_POWER_W:
                            scaled_c = c * scale
                            if len(scaled_c) <= dur_max_allowed:
                                calibrated.append(scaled_c)
                            else:
                                calibrated.append(c)
                        else:
                            # Étape 3 : Garde-fou anti-pic (Plafonnement + Étirement temporel)
                            safe_scale = MAX_POWER_W / max_p
                            safe_c = c * safe_scale
                            safe_e = safe_c.sum() / 60.0
                            stretch_factor = math.ceil(e_target / safe_e)
                            
                            stretched_c = np.repeat(safe_c, stretch_factor)
                            
                            if len(stretched_c) <= dur_max_allowed * 1.5:
                                calibrated.append(stretched_c)
                            else:
                                calibrated.append(c) # Sécurité finale
        # On ne remplace les cycles que si le calibrage a trouvé des solutions valides
        if calibrated:
            cycles = calibrated

    # Garde-fou final : élimine les cycles dont la durée dépasse 1 journée
    # (peut arriver si stretch_factor est très grand sur une faible énergie cible)
    cycles = [c for c in cycles if len(c) <= MINUTES_DAY]
    if not cycles:
        return []

    return cycles


# ── Lecture brute des CSV — cachée indéfiniment par session ─────────────────
# Séparée du calibrage pour éviter de relire 1568 fichiers à chaque interaction.
@st.cache_resource(show_spinner="Chargement des cycles (une seule fois)…")
def _load_raw_cycles() -> tuple:
    """Lit les CSV une seule fois. Retourne (all_cycles, source_map)."""
    global CYCLE_SOURCE
    
    if not hasattr(CYCLES_FINAUX_DIR, 'exists') or not CYCLES_FINAUX_DIR.exists():
        st.warning("Le dossier des cycles est introuvable sur le serveur.")
        return {}, {}
    csv_files = sorted(CYCLES_FINAUX_DIR.glob("*.csv"))
    source_map = {}
    all_cycles = {}
    for csv_file in csv_files:
        if csv_file.name.startswith("_"):
            continue
        stem = csv_file.stem
        if "_cycle" not in stem:
            continue
        prefix, _ = stem.rsplit("_cycle", 1)
        usage, source = None, None
        for kw in ["LPG", "SmartHouse", "REFIT"]:
            tag = f"_{kw}"
            if tag in prefix:
                idx    = prefix.index(tag)
                usage  = prefix[:idx]
                source = prefix[idx+1:]
                break
        if not usage:
            continue
        try:
            df  = pd.read_csv(csv_file)
            pw  = df["powerInW"].values.astype(float)
            if len(pw) < 2:
                continue
            max_w  = MAX_CYCLE_MEAN_W.get(usage)
            active = pw[pw > 0]
            if max_w and len(active) > 0 and active.mean() > max_w:
                continue
            peak = float(pw.max())
            all_cycles.setdefault(usage, []).append((len(pw), pw, source, peak))
            source_map[usage] = source
        except Exception as e:
            pass
    CYCLE_SOURCE = source_map.copy()
    return all_cycles, source_map


def load_all_cycles(dur_overrides: dict = None,
                    peak_overrides: dict = None,
                    cat_overrides: dict = None) -> dict:
    """Filtre et calibre les cycles bruts selon les paramètres UI.
    La lecture CSV est mise en cache — seul le calibrage est recalculé.
    """
    global CYCLE_SOURCE
    if dur_overrides  is None: dur_overrides  = {}
    if peak_overrides is None: peak_overrides = {}
    if cat_overrides  is None: cat_overrides  = {}
    all_cycles, source_map = _load_raw_cycles()
    if not all_cycles:
        return {}
    CYCLE_SOURCE = source_map.copy()
    # Durée minimale physique par usage — protège contre les dur_overrides
    # trop restrictifs issus du profiler ELECDOM (seuil de détection trop bas
    # qui ne capture que les courtes phases actives d'un cycle long).
    DURATION_FLOOR = {
        "washing_machine": [45, 120],
        "dishwasher":      [45, 150],
        "dryer":           [30,  90],
        "oven":            [20,  90],
    }

    result = {}
    for usage, cycle_list in all_cycles.items():
        dur_f  = dur_overrides.get(usage,  CYCLE_DURATION_FILTER.get(usage))
        peak_f = peak_overrides.get(usage)

        # Si le filtre issu du profiler est plus restrictif que le plancher
        # physique, on élargit silencieusement pour ne pas perdre les cycles.
        floor = DURATION_FLOOR.get(usage)
        if dur_f and floor:
            dur_f = [min(dur_f[0], floor[0]), max(dur_f[1], floor[1])]

        def passes(dur, pw, src, peak):
            if dur_f  and not (dur_f[0]  <= dur  <= dur_f[1]):  return False
            if peak_f and not (peak_f[0] <= peak <= peak_f[1]): return False
            return True
        in_range = [pw for dur, pw, src, peak in cycle_list if passes(dur, pw, src, peak)]
        if in_range:
            selected = in_range
        else:
            # Aucun cycle réel dans la plage filtre — pas de fabrication par
            # interpolation (physiquement discutable). On garde tous les
            # cycles bruts disponibles plutôt que d'étirer un cycle hors-plage.
            selected = [pw for _, pw, _, _ in cycle_list]

        cat = cat_overrides.get(usage, {})
        
        if in_range:
            selected = in_range
            result[usage] = apply_cycle_calibration(selected, cat, usage)
        else:
            # Fallback "Meilleur cycle" avec condition de durée
            target_dur = (dur_f[0] + dur_f[1]) / 2.0 if dur_f else 0
            if target_dur > 0 and cycle_list:
                best = min(cycle_list, key=lambda c: abs(c[0] - target_dur))
                selected = [best[1]]  # On ne garde que l'array de puissance
            else:
                selected = [pw for _, pw, _, _ in cycle_list]
            
            cat_fallback = cat.copy()
            
            # ── CORRECTION : Frontière des 10 minutes ──
            # Si c'est un usage très court (ex: Micro-Onde), on supprime la cible
            # énergétique pour forcer l'utilisation du cycle brut une seule fois.
            # Si c'est un usage long (ex: Four), on GARDE la cible énergétique 
            # pour que l'entonnoir le répète et atteigne la bonne consommation.
            if target_dur < 10:
                cat_fallback.pop("energy_target_wh", None)
            
            result[usage] = apply_cycle_calibration(selected, cat_fallback, usage)

    return result



def is_vacation(date: datetime.date, vacances: list) -> bool:

    year = date.year
    for start_md, end_md in vacances:
        try:
            d_start = datetime.date(year, int(start_md[:2]), int(start_md[3:]))
            d_end   = datetime.date(year, int(end_md[:2]),   int(end_md[3:]))
            if d_start <= date <= d_end:
                return True
        except ValueError:
            pass
    return False

def occupation_mask(date: datetime.date, members: list, day_type: str, weekend_params: dict = None) -> np.ndarray:
    """
    Gestion unifiée de la présence : utilise le profil mesuré (semaine ou weekend).
    Si le profil weekend est vide/absent, on utilise le profil semaine par défaut.
    """
    mask = np.zeros(MINUTES_DAY, dtype=bool)
    if day_type == "vacances":
        return mask

    for m in members:
        present = np.ones(MINUTES_DAY, dtype=bool)
        
        # 1. Sélection de la clé de présence (priorité weekend, fallback semaine)
        pres_key = "_presence_weekend" if day_type == "weekend" else "_presence_semaine"
        pres_measured_raw = m.get(pres_key)
        
        # Fallback intelligent : si pas de données weekend, on prend la semaine
        if not pres_measured_raw and day_type == "weekend":
            pres_measured_raw = m.get("_presence_semaine")

        if pres_measured_raw:
            pres_measured = {int(k): v for k, v in pres_measured_raw.items()}
            for h in range(24):
                if random.random() > pres_measured.get(h, 1.0):
                    present[h*60:(h+1)*60] = False
        else:
            # Fallback catalogue (si aucune donnée mesurée)
            pres_profil = m.get("presence_profil", "Personnalisé")
            proba_h = PRESENCE_PROFILES.get(pres_profil, [1.0]*24)
            for h in range(24):
                if random.random() > proba_h[h]:
                    present[h*60:(h+1)*60] = False

        mask |= present

    return mask


def draw_start_minute(h_pic: float, sigma_h: float,
                      presence_mask: np.ndarray,
                      cycle_duration: int,
                      max_tries: int = 50) -> int | None:
    mu    = int(h_pic * 60)
    sigma = int(sigma_h * 60)
    for _ in range(max_tries):
        t = int(np.random.normal(mu, sigma))
        # Évite un pic artificiel à 23h59 si un cycle est tronqué net.
        t = max(0, min(MINUTES_DAY - 1, t))
        if presence_mask[t]:
            return t
    return None

def get_composite_weights(usage: str, members: list, day_type: str) -> list | None:
    """
    Calcule la distribution horaire composite pour un usage selon les profils
    EDT des membres du foyer.
    - Weekend : pas de données EDT → retourne None (fallback catalogue)
    - Semaine : moyenne des distributions EDT des membres avec profil EDT actif
    - Si aucun membre avec profil EDT → retourne None (fallback catalogue)
    """
    if day_type == "weekend":
        return None   # pas de données EDT weekend
    weights_list = []
    for m in members:
        profil = m.get("presence_profil", "Personnalisé")
        if profil in HOURLY_WEIGHTS_INSEE:
            w = HOURLY_WEIGHTS_INSEE[profil].get(usage)
            if w:
                weights_list.append(w)
    if not weights_list:
        return None
    # Moyenne pondérée (1 membre = 1 poids)
    arr = np.mean([np.array(w) for w in weights_list], axis=0)
    total = arr.sum()
    return (arr / total).tolist() if total > 0 else None


def simulate_day(day_type: str,

                 members: list,
                 catalogue_override: dict,
                 cycles_data: dict,
                 forced_events: list,
                 lights_scale: float,
                 bruit_fond_W: float,
                 vacances_periods: list,
                 date: datetime.date,
                 temp_c: float = None,
                 weekend_params: dict = None,
                 age_factors_override: dict = None,
                 liaisons_override: list = None,
                 carry_in: dict = None) -> tuple[np.ndarray, dict, dict]:
    """

    Simule une journée complète avec facteurs saisonniers et météo.
    temp_c : température moyenne du jour (None = pas de données météo).
    carry_in : {usage: array} — fin de cycle entamé la veille à 23h59,
               poursuivi en début de cette journée (continuité 24h->00h).
    Retourne (power_array [1440], contributions {usage: array [1440]},
              carry_out {usage: array} pour le jour suivant).
    """

    season = get_season(date)
    power      = np.zeros(MINUTES_DAY)
    contrib    = {}
    carry_in   = carry_in or {}
    carry_out  = {}
    # Masque de présence
    presence = occupation_mask(date, members, day_type, weekend_params)
    anyone_home = presence.any()

# ── Occupation par membre [n_membres × 1440] ──────────────────────────────
    n_membres   = max(len(members), 1)
    member_busy = np.zeros((n_membres, MINUTES_DAY), dtype=bool)
    
    # Occupation de l'appareil (anti-superposition) ───────────────
    # Couvre CATALOGUE + catalogue_override (usages injectés par le pipeline)
    _all_usages = set(CATALOGUE) | set(catalogue_override)
    machine_busy = {u: np.zeros(MINUTES_DAY, dtype=bool) for u in _all_usages}

    # ── Continuité 24h->00h : injecter la fin du cycle entamé hier ──────────
    for usage, seg in carry_in.items():
        n = min(len(seg), MINUTES_DAY)
        if n <= 0:
            continue
        power[:n] += seg[:n]
        contrib.setdefault(usage, np.zeros(MINUTES_DAY))
        contrib[usage][:n] += seg[:n]
        if usage in machine_busy:
            machine_busy[usage][:n] = True
            
        # ── CORRECTION : Si le cycle/gap de la veille dure encore plus 
        # de 24h, on le repousse à nouveau vers la journée de demain.
        if len(seg) > MINUTES_DAY:
            carry_out[usage] = seg[MINUTES_DAY:]

    def assign_member(usage, t_start, t_end):

        """

        Assigne un usage immobilisant à un membre libre sur ce créneau.
        Retourne l'index du membre assigné, ou None si tous occupés.
        """

        immo = catalogue_override.get(usage, {}).get(
            "immobilisation",
            IMMOBILISATION_DEFAULT.get(usage, 0.0)
        )
        if immo == 0.0 or n_membres == 0:
            return None   # usage passif, pas d'assignation
        # Cherche un membre libre sur le créneau
        free = [i for i in range(n_membres)
                if not member_busy[i, t_start:t_end].any()]
        if not free:
            return None   # tous occupés → on ne place pas
        chosen = random.choice(free)
        # Marque comme occupé proportionnellement au taux d'immobilisation
        if random.random() < immo:
            member_busy[chosen, t_start:t_end] = True
        return chosen

    # ── Bruit de fond constant (box, VMC, veilles génériques) ──────────────
    # N'inclut PAS les veilles spécifiques par appareil (TV, HiFi, PC)
    # qui sont dans puissance_fond — évite le double comptage avec bruit_fond_W
    fond = np.full(MINUTES_DAY, bruit_fond_W)
    contrib["Bruit_fond"] = fond.copy()
    power += fond

    # ── Froid (Frigo & Congélateur) : cycles enchaînés en continu ──────────
    for cold_app in ["fridge", "freezer"]:
        cold_cat = catalogue_override.get(cold_app, {})
        if cold_cat.get("active", True) and cold_app in cycles_data and cycles_data[cold_app]:
            cold_arr      = np.zeros(MINUTES_DAY)
            gap_min       = cold_cat.get("gap_off_min", 60)
            gap_max       = cold_cat.get("gap_off_max", 90)

            cycle_moyen_dur = int(np.mean([len(c) for c in cycles_data[cold_app]]))
            t = random.randint(0, cycle_moyen_dur + gap_max)

            while t < MINUTES_DAY:
                cycle = random.choice(cycles_data[cold_app]).copy().astype(float)
                dur   = len(cycle)
                t_end = min(t + dur, MINUTES_DAY)
                cold_arr[t:t_end] += cycle[:t_end - t]
                t = t_end + random.randint(gap_min, gap_max)
                
            contrib[cold_app] = cold_arr
            power += cold_arr

    # ── Éclairage — puissance continue basée sur présence + ensoleillement ──
    lights_cat = catalogue_override.get("lights", {})
    if lights_cat.get("active", True):
        lights_arr    = np.zeros(MINUTES_DAY)
        lights_w_base = lights_cat.get("lights_w_per_member", LIGHTS_W_PER_MEMBER)
        for t in range(MINUTES_DAY):
            if not presence[t]:
                continue   # personne présent → pas d'éclairage
            dl = daylight_factor(t, season)
            if dl < 0.01:
                continue   # plein jour → pas d'éclairage
            # Puissance proportionnelle au facteur d'ensoleillement
            # lights_scale lisse la puissance selon la taille du foyer
            lights_arr[t] = lights_w_base * lights_scale * dl
        contrib["lights"] = lights_arr
        power += lights_arr
    if day_type == "vacances":
        # Uniquement bruit de fond + frigo + éclairage minimal
        return power, contrib, carry_out

    # ── Helpers internes ─────────────────────────────────────────────────────

    def compute_age_factor(usage):

        """Calcule le facteur âge mixte pour un usage."""

        af_ov = age_factors_override or {}
        factors_all = []
        for m in members:
            age = m.get("age_group", "Adulte actif (18-65 ans)")
            if age in af_ov:
                af = float(af_ov[age].get(usage, 1.0))
            else:
                af = float(AGE_FACTORS.get(age, {}).get(usage, 1.0))
            factors_all.append(af)
        n = max(len(factors_all), 1)
        positives = [af - 1.0 for af in factors_all if af > 1.0]
        if positives:
            return 1.0 + sum(positives) / (n ** 0.5)
        return max(factors_all) if factors_all else 1.0

    def compute_prob(usage, cat):

        """Calcule la probabilité finale pour un usage."""

        prob_key = f"prob_{day_type}"
        prob = cat.get(prob_key, cat.get("prob_semaine", 0))
        s_factor = seasonal_factor(usage, season)
        if temp_c is not None and not np.isnan(float(temp_c)):
            m_factor = meteo_factor(usage, temp_c)
            combined = 0.3 * s_factor + 0.7 * m_factor
        else:
            combined = s_factor
        prob = min(1.0, prob * combined * compute_age_factor(usage))
        return prob

    def place_cycle(usage, t_start, scale=1.0):

        """Place un cycle sur la courbe à t_start.
        Si le cycle dépasse minuit, la partie restante est stockée dans
        carry_out[usage] pour être poursuivie en début de journée suivante
        (continuité 24h->00h, pas de troncature silencieuse).
        """

        if usage not in cycles_data or not cycles_data[usage]:
            return None
        cycle  = random.choice(cycles_data[usage])
        dur    = len(cycle)
        t_end  = min(t_start + dur, MINUTES_DAY)
        seg    = cycle[:t_end - t_start] * scale
        power[t_start:t_end] += seg
        contrib.setdefault(usage, np.zeros(MINUTES_DAY))
        contrib[usage][t_start:t_end] += seg
        assign_member(usage, t_start, t_end)
        if t_start + dur > MINUTES_DAY:
            reste = (cycle[t_end - t_start:] * scale)
            carry_out[usage] = carry_out.get(usage, np.zeros(0))
            # Combine si plusieurs cycles débordent le même jour (rare)
            if len(carry_out[usage]) < len(reste):
                pad = np.zeros(len(reste))
                pad[:len(carry_out[usage])] = carry_out[usage]
                carry_out[usage] = pad
            carry_out[usage][:len(reste)] += reste
        return t_end

    def try_event(usage, cat, scale=1.0):
        if not cat.get("active", True):
            return []
        if usage not in cycles_data or not cycles_data[usage]:
            return []
        if cat.get("continu", False):
            return []
        if cat.get("besoin_presence", True) and not anyone_home:
            return []

        # ── Logique probabiliste : prob + max_uses séparé semaine/weekend ──
        prob = compute_prob(usage, cat)
        if not cat.get("force_no_binomial", False) and usage in PERSONAL_USAGES:
            n_equiv = compute_equiv_persons(members, usage, age_factors_override)
            prob = min(1.0, 1.0 - (1.0 - prob) ** n_equiv)
        if random.random() > prob:
            return []
        # max_uses et decay_probs séparés semaine/weekend
        if day_type == "weekend":
            max_uses   = cat.get("max_uses_weekend",   cat.get("max_uses", 1))
            decay_probs = cat.get("decay_probs_weekend", cat.get("decay_probs", []))
        else:
            max_uses   = cat.get("max_uses_semaine",   cat.get("max_uses", 1))
            decay_probs = cat.get("decay_probs_semaine", cat.get("decay_probs", []))
        n_cycles = 1
        for k, dp in enumerate(decay_probs):
            if n_cycles >= max_uses or random.random() > dp:
                break
            n_cycles += 1

        if n_cycles == 0:
            return []

        placed_ends = []
        for _ in range(n_cycles):
            cycle = random.choice(cycles_data[usage])
            dur   = len(cycle)
            if dur >= MINUTES_DAY:
                continue

            weights = cat.get(f"hourly_weights_{day_type}")
            if not weights:
                weights = get_composite_weights(usage, members, day_type)
            if not weights:
                weights = (HOURLY_WEIGHTS_WEEKEND if day_type == "weekend"
                           else HOURLY_WEIGHTS_SEMAINE).get(usage)

            h_pic_key  = "semaine_h_pic" if day_type == "semaine" else "weekend_h_pic"
            h_pic_base = cat.get(h_pic_key) or cat.get("heure_pic", 12)
            h_pic  = seasonal_h_pic(usage, season, h_pic_base)
            sigma  = cat.get("sigma_h", 2)

            avail_mask = ~machine_busy[usage]
            if cat.get("besoin_presence", True):
                avail_mask &= presence

            if weights:
                t_start = draw_start_density(weights, avail_mask, dur)
            else:
                t_start = draw_start_minute(h_pic, sigma + 0.5, avail_mask, dur)

            if t_start is None:
                continue

            t_end = place_cycle(usage, t_start, scale)
            if t_end:
                machine_busy[usage][t_start:t_end] = True
                placed_ends.append(t_end)

        return placed_ends

    # ── Classification usages ─────────────────────────────────────────────────
    DEPENDANTS = set()
    _lias = liaisons_override if liaisons_override else LIAISONS_DEFAULT
    for l in _lias:
        if l.get("active", True):
            src = l["source"]
            cat_src = catalogue_override.get(src, {})
            # Règle Intelligente : La cible devient dépendante UNIQUEMENT si la source
            # est activée ET a une probabilité > 0 de tourner dans cette maison.
            prob_s = cat_src.get("prob_semaine", 0)
            prob_w = cat_src.get("prob_weekend", 0)
            if cat_src.get("active", True) and (prob_s > 0 or prob_w > 0):
                DEPENDANTS.add(l["cible"])

    def eligible_members(usage, cat):

        """

        Retourne la liste des membres qui peuvent utiliser cet usage.
        Si age_requis vide → tous les membres.
        """

        age_req = cat.get("age_requis", [])
        if not age_req:
            return members
        return [m for m in members
                if m.get("age_group", "Adulte actif (18-65 ans)") in age_req]

    def try_personal_events(usage, cat, scale=1.0):
        """Usage personnel (TV, gaming, PC…).
        Tirage du nombre total de cycles selon une moyenne de Poisson
        (mean_cycles_semaine / mean_cycles_weekend) si disponible,
        sinon fallback sur l'ancienne logique (probabilité individuelle + max_uses)."""
        if not cat.get("active", True):
            return
        if usage not in cycles_data or not cycles_data[usage]:
            return
        if cat.get("continu", False):
            return
        if cat.get("besoin_presence", True) and not anyone_home:
            return

        eligible = eligible_members(usage, cat)
        if not eligible:
            return

        # ----- FALLBACK : ancienne méthode (probabilité individuelle + max_uses) -----
        af_ov = age_factors_override or {}
        h_pic_key = "semaine_h_pic" if day_type == "semaine" else "weekend_h_pic"
        h_pic_base = cat.get(h_pic_key) or cat.get("heure_pic", 12)
        h_pic = seasonal_h_pic(usage, season, h_pic_base)
        sigma = cat.get("sigma_h", 2)

        s_factor = seasonal_factor(usage, season)
        if temp_c is not None and not np.isnan(float(temp_c)):
            m_factor = meteo_factor(usage, temp_c)
            combined = 0.3 * s_factor + 0.7 * m_factor
        else:
            combined = s_factor

        prob_key = f"prob_{day_type}"
        prob_base = cat.get(prob_key, cat.get("prob_semaine", 0))

        # max_uses et decay_probs selon jour type
        if day_type == "weekend":
            max_uses = cat.get("max_uses_weekend", cat.get("max_uses", 1))
            decay_probs = cat.get("decay_probs_weekend", cat.get("decay_probs", []))
        else:
            max_uses = cat.get("max_uses_semaine", cat.get("max_uses", 1))
            decay_probs = cat.get("decay_probs_semaine", cat.get("decay_probs", []))

        for m_idx, m in enumerate(eligible):
            age = m.get("age_group", "Adulte actif (18-65 ans)")
            if age in af_ov:
                af = float(af_ov[age].get(usage, 1.0))
            else:
                af = float(AGE_FACTORS.get(age, {}).get(usage, 1.0))
            prob_indiv = min(1.0, prob_base * combined * af)
            current_prob = prob_indiv

            for use_idx in range(max_uses):
                if random.random() > current_prob:
                    break

                cycle = random.choice(cycles_data[usage])
                dur = len(cycle)

                m_profil = m.get("presence_profil", "Personnalisé")
                m_weights = None
                if day_type == "weekend":
                    m_weights = cat.get("hourly_weights_weekend")
                if not m_weights:
                    m_weights = cat.get("hourly_weights_semaine") or cat.get("hourly_weights_vacances")
                if not m_weights:
                    if day_type == "semaine" and m_profil in HOURLY_WEIGHTS_INSEE:
                        m_weights = HOURLY_WEIGHTS_INSEE[m_profil].get(usage)
                if not m_weights:
                    m_weights = (HOURLY_WEIGHTS_WEEKEND if day_type == "weekend"
                                 else HOURLY_WEIGHTS_SEMAINE).get(usage)

                # Masque double : membre et machine libres
                avail_mask = presence & ~member_busy[m_idx] & ~machine_busy[usage]

                if m_weights:
                    t_start = draw_start_density(m_weights, avail_mask, dur)
                else:
                    t_start = draw_start_minute(h_pic, sigma + 0.5, avail_mask, dur)

                if t_start is None:
                    break
                if usage == "lights" and not presence[t_start]:
                    break

                t_end = min(t_start + dur, MINUTES_DAY)
                seg = cycle[:t_end - t_start] * scale
                power[t_start:t_end] += seg
                contrib.setdefault(usage, np.zeros(MINUTES_DAY))
                contrib[usage][t_start:t_end] += seg

                # Immobilisation
                member_busy[m_idx, t_start:t_end] = True
                machine_busy[usage][t_start:t_end] = True

                # ── CORRECTION : Continuité 24h->00h pour les usages personnels ──
                # Si le film ou l'ordinateur tourne encore après 23h59, 
                # on stocke la fin pour l'injecter le lendemain matin.
                if t_start + dur > MINUTES_DAY:
                    reste = cycle[t_end - t_start:] * scale
                    carry_out[usage] = carry_out.get(usage, np.zeros(0))
                    if len(carry_out[usage]) < len(reste):
                        pad = np.zeros(len(reste))
                        pad[:len(carry_out[usage])] = carry_out[usage]
                        carry_out[usage] = pad
                    carry_out[usage][:len(reste)] += reste

                if use_idx < len(decay_probs):
                    current_prob = decay_probs[use_idx]
                else:
                    break

    # ── Liaisons logiques (dynamiques depuis LIAISONS_DEFAULT + overrides UI) ──
    # Récupère les liaisons actives depuis la session (modifiables dans l'UI)
    liaisons_actives = liaisons_override if liaisons_override else LIAISONS_DEFAULT

    # Collecte les sources uniques → déclenche chaque source une seule fois
    sources_a_declencher = list({l["source"] for l in liaisons_actives
                                  if l.get("active", True)})
    for seq_app in ["oven", "washing_machine", "hob", "coffee_machine"]:
        if seq_app not in sources_a_declencher:
            sources_a_declencher.append(seq_app)

    triggered = set()

    # Déclenche chaque source
    t_ends = {}   
    for src in sources_a_declencher:
        src_cat = catalogue_override.get(src, {})
        placed = try_event(src, src_cat)
        if placed:
            triggered.add(src)
            t_ends[src] = placed

    # Déclenche les cibles selon les liaisons
    for liaison in liaisons_actives:
        if not liaison.get("active", True): continue
        src = liaison["source"]
        cible = liaison["cible"]
        if src not in t_ends: continue
        
        # Pour CHAQUE fois que la source a tourné
        for t_src_end in t_ends[src]:
            if random.random() > liaison.get("prob", 1.0): continue

            d_min, d_max = liaison.get("delai_min", 0), liaison.get("delai_max", 0)
            delay = d_min if d_min == d_max else random.randint(min(d_min, d_max), max(d_min, d_max))
            t_cible = t_src_end + delay
            
            cible_cat = catalogue_override.get(cible, {})
            if (t_cible >= 0 and t_cible < MINUTES_DAY and cible_cat.get("active", True) and cycles_data.get(cible)):
                
                # Attend que la machine cible soit disponible
                while t_cible < MINUTES_DAY and machine_busy[cible][t_cible]:
                    t_cible += 1
                
                # Attend la présence si demandé
                if liaison.get("attend_presence", False):
                    while t_cible < MINUTES_DAY and not presence[t_cible]:
                        t_cible += 1
                        
                if t_cible < MINUTES_DAY:
                    t_end_c = place_cycle(cible, t_cible)
                    if t_end_c:
                        machine_busy[cible][t_cible:t_end_c] = True
                        triggered.add(cible)

    # ── 5. Événements indépendants (tous les usages non déclenchés) ───────────
    # lights géré séparément (courbe continue), fridge géré en boucle
    SEQUENCED = {"washing_machine", "oven", "hob", "coffee_machine",
                 "lights", "fridge"}
    for usage, cat in catalogue_override.items():
        if usage in triggered or usage in SEQUENCED:
            continue
        if usage in DEPENDANTS:
            continue
        if usage in PERSONAL_USAGES:
            # Usages personnels : N tirages indépendants par membre
            try_personal_events(usage, cat)
        else:
            # Usages foyer : 1 seul événement
            try_event(usage, cat)
    # ── Événements forcés ────────────────────────────────────────────────────
    # Deux modes :
    #   mode='fixed'    : heure fixe (comportement historique)
    #   mode='programme': usage programmé hors présence, tirage selon
    #                     hourly_weights + prob de déclenchement
    for ev in forced_events:
        usage = ev.get("usage")
        if usage not in cycles_data or not cycles_data[usage]:
            continue
        mode = ev.get("mode", "fixed")
        prob_ev = float(ev.get("prob", 1.0))
        if random.random() > prob_ev:
            continue   # pas déclenché ce jour

        cycle = random.choice(cycles_data[usage])
        dur   = len(cycle)

        if mode == "programme":
            # Tirage selon distribution horaire mesurée, sans contrainte présence
            weights = ev.get("hourly_weights")
            if weights:
                w = np.array(weights, dtype=float)
                s = w.sum()
                if s > 0:
                    w = w / s
                else:
                    w = np.ones(24) / 24
            else:
                w = np.ones(24) / 24
            # Masque : machine libre uniquement (pas de contrainte présence)
            avail = ~machine_busy.get(usage, np.zeros(MINUTES_DAY, dtype=bool))
            t = draw_start_density(w.tolist(), avail, dur)
            if t is None:
                # Fallback : première minute libre
                free = np.where(avail)[0]
                t = int(free[0]) if len(free) > 0 else 0
        else:
            # Mode fixe historique
            h_start = ev.get("heure", 12)
            t = int(h_start * 60)
            t = max(0, min(MINUTES_DAY - 1, t))

        t_end = min(t + dur, MINUTES_DAY)
        seg   = cycle[:t_end - t]
        power[t:t_end]          += seg
        contrib.setdefault(usage, np.zeros(MINUTES_DAY))
        contrib[usage][t:t_end] += seg
        if usage in machine_busy:
            machine_busy[usage][t:t_end] = True

        # Continuité 24h->00h pour les événements programmés (cf place_cycle).
        # Si le cycle déborde sur le jour suivant, on sauvegarde le reste 
        # dans carry_out pour qu'il soit injecté à 00h00 le lendemain.
        if t + dur > MINUTES_DAY:
            reste = cycle[t_end - t:]
            carry_out[usage] = carry_out.get(usage, np.zeros(0))
            # On ajuste la taille du tableau si plusieurs cycles débordent (rare mais robuste)
            if len(carry_out[usage]) < len(reste):
                pad = np.zeros(len(reste))
                pad[:len(carry_out[usage])] = carry_out[usage]
                carry_out[usage] = pad
            carry_out[usage][:len(reste)] += reste
    return power, contrib, carry_out

def classify_day(date: datetime.date, vacances_periods: list) -> str:

    """Retourne 'vacances', 'weekend' ou 'semaine' pour une date donnée."""

    if is_vacation(date, vacances_periods):
        return "vacances"
    if date.weekday() >= 5:   # samedi=5, dimanche=6
        return "weekend"
    return "semaine"

def run_simulation(date_start: datetime.date,

                   date_end: datetime.date,
                   members: list,
                   catalogue_override: dict,
                   cycles_data: dict,
                   forced_events: list,
                   lights_scale: float,
                   bruit_fond_W: float,
                   vacances_periods: list,
                   meteo_data: dict = None,
                   weekend_params: dict = None,
                   age_factors_override: dict = None,
) -> dict:
    """

    Lance la simulation jour par jour entre date_start et date_end.
    Chaque jour est classé semaine / weekend / vacances.
    Retourne les arrays moyens + par saison pour chaque type de journée.
    """

    results  = {"semaine": [], "weekend": [], "vacances": []}
    contribs = {"semaine": {}, "weekend": {}, "vacances": {}}
    # Aussi par saison (hors vacances)
    saisons      = ["hiver", "printemps", "été", "automne"]
    res_saison   = {s: [] for s in saisons}
    cont_saison  = {s: {} for s in saisons}
    # Par saison × type pour stats détaillées
    res_saison_dtype  = {f"{s}_{d}": [] for s in saisons
                         for d in ["semaine", "weekend"]}
    # ── Construction de la liste des journées à simuler ──────────────────
    n_days = (date_end - date_start).days + 1
    liaisons_now = st.session_state.get("liaisons") if "streamlit" in dir() else None
    job_list = []
    current = date_start
    for _ in range(n_days):
        day_type     = classify_day(current, vacances_periods)
        forced_today = [ev for ev in forced_events
                        if day_type in ev.get("types", [])]
        temp_c = (meteo_data or {}).get(current)
        job_list.append((current, day_type, forced_today, temp_c))
        current += datetime.timedelta(days=1)

    # ── Simulation parallèle par chunks de jours ─────────────────────────
    # On découpe en chunks de 30 jours — chaque chunk est simulé dans un
    # thread séparé. Limité aux threads (pas process) pour éviter les
    # problèmes de pickle avec les cycles numpy.
    from concurrent.futures import ThreadPoolExecutor
    CHUNK = 30
    chunks = [job_list[i:i+CHUNK] for i in range(0, len(job_list), CHUNK)]

    def simulate_chunk(jobs):
        chunk_results = []
        # Continuité 24h->00h : carry_in du 1er jour du chunk = vide.
        # Léger artefact aux frontières de chunk (tous les 30 jours),
        # négligeable sur une moyenne annuelle.
        carry = {}
        for date, day_type, forced_today, temp_c in jobs:
            p, c, carry = simulate_day(
                day_type, members, catalogue_override,
                cycles_data, forced_today, lights_scale,
                bruit_fond_W, vacances_periods, date,
                temp_c=temp_c,
                weekend_params=weekend_params,
                age_factors_override=age_factors_override,
                liaisons_override=liaisons_now,
                carry_in=carry,
            )
            chunk_results.append((date, day_type, p, c))
        return chunk_results

    all_day_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(simulate_chunk, chunk) for chunk in chunks]
        for future in futures:
            all_day_results.extend(future.result())

    # ── Agrégation ───────────────────────────────────────────────────────
    for date, day_type, p, c in all_day_results:
        season = get_season(date)
        results[day_type].append(p)
        all_known_dt = set(c.keys()) | set(contribs[day_type].keys())
        for k in all_known_dt:
            v = c.get(k, np.zeros(MINUTES_DAY))
            contribs[day_type].setdefault(k, []).append(v)
        if day_type != "vacances":
            res_saison[season].append(p)
            all_known = set(c.keys()) | set(cont_saison[season].keys())
            for k in all_known:
                v = c.get(k, np.zeros(MINUTES_DAY))
                cont_saison[season].setdefault(k, []).append(v)
            key_sd = f"{season}_{day_type}"
            if key_sd in res_saison_dtype:
                res_saison_dtype[key_sd].append(p)
    # Moyennes
    out = {}
    for dtype in ["semaine", "weekend", "vacances"]:
        if results[dtype]:
            arr = np.array(results[dtype])
            out[f"mean_{dtype}"]    = np.mean(arr, axis=0)
            out[f"min_{dtype}"]     = np.percentile(arr,  5, axis=0)
            out[f"max_{dtype}"]     = np.percentile(arr, 95, axis=0)
            out[f"n_{dtype}"]       = len(results[dtype])
            out[f"contrib_{dtype}"] = {
                k: np.mean(v, axis=0)
                for k, v in contribs[dtype].items()
            }
        else:
            out[f"mean_{dtype}"]    = np.zeros(MINUTES_DAY)
            out[f"min_{dtype}"]     = np.zeros(MINUTES_DAY)
            out[f"max_{dtype}"]     = np.zeros(MINUTES_DAY)
            out[f"n_{dtype}"]       = 0
            out[f"contrib_{dtype}"] = {}
    for s in saisons:
        if res_saison[s]:
            arr_s = np.array(res_saison[s])
            out[f"mean_{s}"]    = np.mean(arr_s, axis=0)
            out[f"min_{s}"]     = np.percentile(arr_s,  5, axis=0)
            out[f"max_{s}"]     = np.percentile(arr_s, 95, axis=0)
            out[f"n_{s}"]       = len(res_saison[s])
            out[f"contrib_{s}"] = {
                k: np.mean(v, axis=0)
                for k, v in cont_saison[s].items()
            }
        else:
            out[f"mean_{s}"] = np.zeros(MINUTES_DAY)
            out[f"min_{s}"]  = np.zeros(MINUTES_DAY)
            out[f"max_{s}"]  = np.zeros(MINUTES_DAY)
            out[f"n_{s}"]    = 0
            out[f"contrib_{s}"] = {}
    # Stats par saison × type de jour
    for key, arr in res_saison_dtype.items():
        if arr:
            out[f"mean_{key}"] = np.mean(arr, axis=0)
            out[f"n_{key}"]    = len(arr)
        else:
            out[f"mean_{key}"] = np.zeros(MINUTES_DAY)
            out[f"n_{key}"]    = 0
    return out

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION

# ═══════════════════════════════════════════════════════════════════════════════

def make_time_axis():

    return [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]

def plot_curves(sim_results: dict, selected_types: list) -> go.Figure:

    """Courbes moyennes + enveloppe pour les types sélectionnés."""

    fig   = go.Figure()
    time  = make_time_axis()
    colors = {
        "semaine": "#2196F3", "weekend": "#FF9800", "vacances": "#4CAF50",
        "hiver": "#90CAF9", "printemps": "#A5D6A7",
        "été": "#FFCC80", "automne": "#FFAB91",
    }
    labels = {
        "semaine": "Semaine", "weekend": "Weekend", "vacances": "Vacances",
        "hiver": "❄️ Hiver", "printemps": "🌸 Printemps",
        "été": "☀️ Été", "automne": "🍂 Automne",
    }
    for dtype in selected_types:
        mean = sim_results.get(f"mean_{dtype}", np.zeros(MINUTES_DAY))
        lo   = sim_results.get(f"min_{dtype}",  np.zeros(MINUTES_DAY))
        hi   = sim_results.get(f"max_{dtype}",  np.zeros(MINUTES_DAY))
        col  = colors[dtype]
        fig.add_trace(go.Scatter(
            x=time, y=hi.tolist(), mode="lines",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        # Convertit #RRGGBB en rgba pour l'enveloppe
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        b = int(col[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.15)"
        fig.add_trace(go.Scatter(
            x=time, y=lo.tolist(), mode="lines",
            fill="tonexty", fillcolor=fill_color,
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=time, y=mean.tolist(), mode="lines",
            name=labels[dtype],
            line=dict(color=col, width=2),
        ))
    fig.update_layout(
        title="Courbe de charge moyenne (enveloppe P5-P95)",
        xaxis_title="Heure", yaxis_title="Puissance (W)",
        xaxis=dict(tickmode="array",
                   tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
                   ticktext=[f"{h:02d}h" for h in range(0, 24, 2)]),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=420,
        margin=dict(t=50, b=60),
    )
    return fig

def plot_stackplot(sim_results: dict, dtype: str,

                   catalogue_override: dict) -> go.Figure:
    """Stackplot de contributions par usage."""

    contrib = sim_results.get(f"contrib_{dtype}", {})
    time    = make_time_axis()
    # Calcule l'énergie totale par usage
    totals = {k: v.sum() for k, v in contrib.items()}
    threshold = SEUIL_STACKPLOT_W * MINUTES_DAY / 60   # en Wh
    main_usages = [k for k, e in totals.items() if e >= threshold]
    other_usages = [k for k, e in totals.items() if e < threshold and k != "Bruit_fond"]
    # Agrège "Autre"
    other_arr = sum(contrib[k] for k in other_usages) if other_usages else np.zeros(MINUTES_DAY)
    traces = []
    # Bruit de fond en premier (base)
    if "Bruit_fond" in contrib:
        traces.append(("Bruit_fond", contrib["Bruit_fond"]))
    # Usages principaux triés par énergie décroissante
    main_sorted = sorted(
        [u for u in main_usages if u != "Bruit_fond"],
        key=lambda u: totals[u], reverse=True
    )
    for u in main_sorted:
        traces.append((u, contrib[u]))
    # Autre en dernier
    if other_usages:
        traces.append(("Autre", other_arr))
    fig = go.Figure()
    for u, arr in traces:
        label = CATALOGUE.get(u, {}).get("label", u)
        col   = USAGE_COLORS.get(u, "#90A4AE")
        fig.add_trace(go.Scatter(
            x=time, y=arr.tolist(),
            name=label, mode="lines",
            stackgroup="one",
            fillcolor=col,
            line=dict(color=col, width=0.5),
        ))
    labels_day = {"semaine": "Semaine", "weekend": "Weekend", "vacances": "Vacances"}
    fig.update_layout(
        title=f"Décomposition par usage — {labels_day.get(dtype, dtype)}",
        xaxis_title="Heure", yaxis_title="Puissance (W)",
        xaxis=dict(tickmode="array",
                   tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
                   ticktext=[f"{h:02d}h" for h in range(0, 24, 2)]),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.20, font=dict(size=10)),
        height=450,
        margin=dict(t=50, b=80),
    )
    return fig

def compute_stats(sim_results: dict) -> pd.DataFrame:

    rows = []
    labels_day = {"semaine": "Semaine", "weekend": "Weekend", "vacances": "Vacances"}
    for dtype, label in labels_day.items():
        mean = sim_results.get(f"mean_{dtype}", np.zeros(MINUTES_DAY))
        p95  = sim_results.get(f"max_{dtype}",  mean)
        rows.append({
            "Type de journée":        label,
            "Puissance moy. (W)":     round(float(mean.mean()), 1),
            "Pic moy. (W)":           round(float(mean.max()),  1),
            "Pic P95 (W)":            round(float(p95.max()),   1),
            "Énergie/jour (kWh)":     round(float(mean.sum()) / 60_000, 3),
        })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT

# ═══════════════════════════════════════════════════════════════════════════════

def _render_calendar(date_start: datetime.date,

                     date_end: datetime.date,
                     vacances: list) -> str:
    """Génère un mini-calendrier HTML coloré."""

    colors = {
        "semaine":  "#BBDEFB",
        "weekend":  "#FFE0B2",
        "vacances": "#C8E6C9",
    }
    labels_c = {"semaine": "Semaine", "weekend": "Weekend", "vacances": "Vacances"}
    saison_emoji = {"hiver": "❄️", "printemps": "🌸", "été": "☀️", "automne": "🍂"}
    html = """<style>
    .cal-grid{display:flex;flex-wrap:wrap;gap:3px;font-size:11px;}
    .cal-day{width:32px;height:32px;display:flex;align-items:center;
             justify-content:center;border-radius:4px;font-weight:600;}
    .cal-legend{display:flex;gap:12px;font-size:11px;margin-bottom:6px;}
    .cal-dot{width:12px;height:12px;border-radius:3px;display:inline-block;}
    </style>"""

    html += "<div class='cal-legend'>"
    for k, col in colors.items():
        html += (f"<span><span class='cal-dot' style='background:{col}'></span>"
                 f" {labels_c[k]}</span>")
    html += "</div><div class='cal-grid'>"
    cur = date_start
    while cur <= date_end:
        dt  = classify_day(cur, vacances)
        col = colors[dt]
        s   = get_season(cur)
        tip = f"{cur.strftime('%d/%m')} {saison_emoji[s]}"
        html += (f"<div class='cal-day' style='background:{col}' "
                 f"title='{tip}'>{cur.day}</div>")
        cur += datetime.timedelta(days=1)
    html += "</div>"
    return html

def main():
    st.write("Le code a bien démarré !")
    st.set_page_config(page_title="Simulateur de charge", page_icon="🏠",
                       layout="wide")
    st.title("🏠 Simulateur de courbe de charge résidentielle")
    # Nettoie les anciennes valeurs de poids horaires en session
    # (évite StreamlitValueAboveMaxError si l'interface a changé)
    stale = [k for k in st.session_state if k.startswith("hw_")]
    for k in stale:
        if isinstance(st.session_state[k], float) and st.session_state[k] > 1.0:
            del st.session_state[k]
    # Lecture brute des cycles — mise en cache @st.cache_resource (une seule fois)
    _raw_cycles, _ = _load_raw_cycles()
    # Cycles prêts pour l'affichage UI (sans calibrage énergétique — juste filtre durée)
    _cat_ov_now = st.session_state.get("catalogue_override", {})
    cycles_data = load_all_cycles(
        dur_overrides={u: c["dur_filter"]  for u, c in _cat_ov_now.items()
                       if c.get("dur_filter")},
        peak_overrides={u: c["peak_filter"] for u, c in _cat_ov_now.items()
                        if c.get("peak_filter")},
    )   # cat_overrides omis ici → pas de calibrage énergétique au render
        # Le calibrage complet est fait uniquement au moment de lancer la simulation
    if not _raw_cycles:
        st.warning(f"⚠️ Aucun cycle trouvé dans `{CYCLES_FINAUX_DIR}`. "
                   f"Lance d'abord l'extracteur LPG.")
                   
    tab_foyer, tab_edt, tab_equip, tab_freq, tab_liaisons, tab_sim, tab_res = st.tabs(
        ["🏠 Foyer", "📋 Profils EDT", "⚙️ Équipements", "🔁 Fréquences", "🔗 Liaisons", "📅 Simulation", "📊 Résultats"]
    )

    # ── Onglet Foyer ──────────────────────────────────────────────────────────
    with tab_foyer:
        st.subheader("Composition du foyer")
        col1, col2 = st.columns([1, 2])
        with col1:
            n_members = st.number_input("Nombre de membres", 1, 8, 3, key="n_members")
        members = []
        st.markdown("---")
        for i in range(int(n_members)):
            st.markdown(f"**Membre {i+1}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.selectbox(f"Tranche d'âge",
                                   AGE_GROUPS, index=3,
                                   key=f"age_{i}")
            with c2:
                # Absences semaine
                h_abs_start = st.number_input(
                    f"Absent semaine de (h)", 0.0, 23.5, 8.0,
                    step=0.5, key=f"abs_start_{i}",
                    help="Heure de début d'absence en semaine"
                )
            with c3:
                h_abs_end = st.number_input(
                    f"Absent semaine à (h)", 0.0, 24.0, 17.0,
                    step=0.5, key=f"abs_end_{i}",
                    help="Heure de fin d'absence en semaine"
                )
            # 2e créneau d'absence (pause déjeuner etc.)
            show_2nd = st.checkbox(f"Ajouter un 2e créneau d'absence",
                                   key=f"abs2_check_{i}")
            absences = [(h_abs_start, h_abs_end)] if h_abs_start < h_abs_end else []
            if show_2nd:
                c4, c5 = st.columns(2)
                with c4:
                    h2s = st.number_input(f"2e créneau — de (h)",
                                          0.0, 23.5, 12.0, step=0.5,
                                          key=f"abs2s_{i}")
                with c5:
                    h2e = st.number_input(f"2e créneau — à (h)",
                                          0.0, 24.0, 13.5, step=0.5,
                                          key=f"abs2e_{i}")
                if h2s < h2e:
                    absences.append((h2s, h2e))
            # Profil de présence — par défaut selon la tranche d'âge
            _default_profil = AGE_TO_EEDT_PROFIL.get(age, "Adulte actif")
            _profil_options = list(PRESENCE_PROFILES.keys())
            _age_key        = f"age_pres_{i}"
            _pres_key       = f"pres_profil_{i}"
            # Quand la tranche change → supprime la key pour forcer le bon défaut
            if st.session_state.get(_age_key) != age:
                st.session_state[_age_key] = age
                if _pres_key in st.session_state:
                    del st.session_state[_pres_key]
            # Initialise si absente
            if _pres_key not in st.session_state:
                st.session_state[_pres_key] = _default_profil
            _default_idx = (_profil_options.index(st.session_state[_pres_key])
                            if st.session_state[_pres_key] in _profil_options else 0)
            st.markdown(f"*📍 Présence — Membre {i+1}*")
            presence_profil = st.selectbox(
                "Profil de présence",
                _profil_options,
                index=_default_idx,
                key=_pres_key,
                help="Personnalisé = utilise les créneaux d'absence manuels. "
                     "Autres = présence probabiliste par heure (Enquête EDT), les créneaux manuels sont ignorés."
            )
            if PRESENCE_PROFILES.get(presence_profil) is not None:
                proba_h = PRESENCE_PROFILES[presence_profil]
                df_pres = pd.DataFrame({
                    "Heure":     [f"{h:02d}h" for h in range(24)],
                    "% présent": [round(p*100, 1) for p in proba_h],
                })
                st.bar_chart(df_pres.set_index("Heure"),
                             height=80, use_container_width=True)
                # Modification manuelle des probabilités de présence
                with st.expander("✏️ Modifier les probabilités de présence", expanded=False):
                    st.caption("Probabilité d'être au domicile heure par heure (0 = absent, 1 = toujours présent). "
                               "Les absences manuelles configurées ci-dessus s'appliquent en plus.")
                    cols_p = st.columns(8)
                    custom_proba = []
                    for h in range(24):
                        with cols_p[h % 8]:
                            val = st.number_input(
                                f"{h:02d}h", 0.0, 1.0,
                                float(round(proba_h[h], 2)),
                                step=0.05,
                                key=f"pres_h_{i}_{h}",
                                format="%.2f",
                                label_visibility="visible"
                            )
                            custom_proba.append(val)
                    # Stocke le profil personnalisé
                    PRESENCE_PROFILES[f"custom_{i}"] = custom_proba
                    presence_profil_effective = f"custom_{i}"
            else:
                presence_profil_effective = presence_profil
            members.append({
                "age_group":        age,
                "absences_semaine": absences,
                "presence_profil":  presence_profil_effective
                                    if PRESENCE_PROFILES.get(presence_profil) is not None
                                    else presence_profil,
            })
            st.markdown("---")
        # ── Présence weekend ─────────────────────────────────────────────────
        # Profil unique modifiable, par défaut identique à la semaine.
        # Distribution horaire : 24 poids (0-1) séparés par des virgules,
        # même format que les distributions horaires des usages.
        st.markdown("---")
        st.subheader("📅 Présence weekend")
        st.caption(
            "Probabilité de présence par heure (0h→23h), séparée par des virgules. "
            "Par défaut identique au profil de présence semaine. "
            "Les données mesurées ELECDOM (_presence_weekend) ont priorité si présentes."
        )
        # Valeur par défaut = profil de présence semaine du 1er membre
        _default_we = members[0].get("_presence_weekend") if members else None
        if not _default_we:
            _pp = members[0].get("presence_profil", "Personnalisé") if members else "Personnalisé"
            _default_we = PRESENCE_PROFILES.get(_pp, [1.0]*24)
            _default_we = {h: v for h, v in enumerate(_default_we)}
        _default_we_str = ", ".join(
            str(round(float(_default_we.get(h, _default_we.get(str(h), 1.0))), 3))
            for h in range(24)
        )
        we_key = "presence_weekend_txt"
        if we_key not in st.session_state:
            st.session_state[we_key] = _default_we_str
        we_txt = st.text_input("Présence weekend 0h→23h", key=we_key)
        try:
            we_weights = [float(x.strip()) for x in we_txt.split(",")]
            if len(we_weights) != 24:
                raise ValueError
            we_weights = [min(1.0, max(0.0, w)) for w in we_weights]
        except Exception:
            we_weights = [float(_default_we.get(h, _default_we.get(str(h), 1.0)))
                          for h in range(24)]
            st.caption("⚠️ 24 valeurs (0-1) attendues — profil semaine utilisé.")
        df_we = pd.DataFrame({
            "Heure": [f"{h:02d}h" for h in range(24)],
            "% présent": [round(w*100, 1) for w in we_weights],
        })
        st.bar_chart(df_we.set_index("Heure"), height=80, use_container_width=True)
        # Applique à tous les membres — écrase _presence_weekend (ou la crée)
        we_dict = {h: w for h, w in enumerate(we_weights)}
        for m in members:
            m["_presence_weekend"] = we_dict
        st.session_state["members"] = members

    # ── Onglet Profils EDT ────────────────────────────────────────────────────
    with tab_edt:
        members_edt = st.session_state.get("members", [])
        st.markdown("""

L'**Enquête Emploi du Temps (EDT)** de l'INSEE recueille les carnets d'activité
d'individus sur une journée de semaine. Elle fournit des données observées sur :
les horaires de sommeil, la présence au domicile, la fréquence des activités
domestiques et les distributions horaires d'usage des équipements.
> ⚠️ Les données EDT couvrent **les jours de semaine uniquement**.
> Le weekend conserve le mode standard (distributions élargies, α=0.5).
""")
        if not members_edt:
            st.info("Configurez d'abord les membres dans l'onglet 🏠 Foyer.")
        else:
            for i, m in enumerate(members_edt):
                age   = m.get("age_group", "Adulte actif (18-65 ans)")
                profil= AGE_TO_EEDT_PROFIL.get(age)
                pres  = m.get("presence_profil", "Personnalisé")
                is_edt= profil is not None and pres != "Personnalisé"
                with st.expander(f"Membre {i+1} — {age} {'✅ Mode EDT' if is_edt else '⚙️ Mode personnalisé'}", expanded=False):
                    # Tableau de transparence
                    EDT_USAGES = list(PROB_INSEE.get(age, {}).keys())
                    has_edt    = HOURLY_WEIGHTS_INSEE.get(profil or "", {})
                    rows = []
                    # Présence
                    rows.append({
                        "Paramètre":  "📍 Présence semaine",
                        "Mode":       f"✅ EDT ({pres})" if is_edt else "⚙️ Créneaux manuels",
                        "Valeur":     "Courbe probabiliste" if is_edt else "Créneaux fixes",
                        "Note":       "Probabilité par heure" if is_edt else "Absent entre les créneaux définis",
                    })
                    rows.append({
                        "Paramètre":  "📍 Présence weekend",
                        "Mode":       "⬜ Inactif",
                        "Valeur":     "Mode standard",
                        "Note":       "Pas de données EDT weekend — α=0.5 uniforme",
                    })
                    # Probabilités d'usage
                    prob_edt = PROB_INSEE.get(age, {})
                    for usage, prob in prob_edt.items():
                        lbl = CATALOGUE.get(usage, {}).get("label", usage)
                        rows.append({
                            "Paramètre":  f"⚡ Prob. {lbl}",
                            "Mode":       "✅ EDT (force_no_binomial)" if is_edt else "🔵 Catalogue",
                            "Valeur":     f"{prob*100:.0f}%" if is_edt else f"{CATALOGUE.get(usage,{}).get('prob_semaine',0)*100:.0f}%",
                            "Note":       "Fréquence journalière observée, binomiale désactivée" if is_edt
                                         else "Valeur catalogue, loi binomiale active",
                        })
                    # Distributions horaires
                    for usage in list(has_edt.keys())[:4]:
                        lbl = CATALOGUE.get(usage, {}).get("label", usage)
                        rows.append({
                            "Paramètre":  f"⏱ Distrib. {lbl}",
                            "Mode":       "✅ EDT (semaine)" if is_edt else "🔵 Catalogue",
                            "Valeur":     "Profil observé EDT" if is_edt else "Distribution catalogue",
                            "Note":       "Weekend → mode standard α=0.5",
                        })
                    # Binomiale
                    rows.append({
                        "Paramètre":  "🎲 Loi binomiale",
                        "Mode":       "⚙️ Mixte",
                        "Valeur":     "EDT → désactivée | Autres → active",
                        "Note":       "Usages sans données EDT gardent la loi binomiale",
                    })
                    # Immobilisation
                    rows.append({
                        "Paramètre":  "🔒 Immobilisation membres",
                        "Mode":       "🔵 Catalogue",
                        "Valeur":     "Paramètres catalogue",
                        "Note":       "Non couvert par l'EDT — valeurs manuelles",
                    })
                    # Sélection cycles
                    has_filter = any(u in CYCLE_DURATION_FILTER
                                     for u in EDT_USAGES)
                    rows.append({
                        "Paramètre":  "📏 Durée des cycles",
                        "Mode":       "✅ Filtre actif" if has_filter else "⬜ Inactif",
                        "Valeur":     "Plages [min,max] configurées" if has_filter else "Tous cycles chargés",
                        "Note":       "Configurable dans l'onglet Équipements",
                    })
                    df_status = pd.DataFrame(rows)
                    # Colorise les modes

                    def color_mode(val):

                        if "✅" in val:   return "background-color:#E8F5E9;color:#1B5E20"
                        if "⬜" in val:   return "background-color:#FAFAFA;color:#9E9E9E"
                        if "⚙️" in val:   return "background-color:#FFF3E0;color:#E65100"
                        if "🔵" in val:   return "background-color:#E3F2FD;color:#0D47A1"
                        return ""
                    st.dataframe(
                        df_status.style.map(color_mode, subset=["Mode"]),
                        use_container_width=True,
                        hide_index=True
                    )
                    # Aperçu courbe de présence si EDT actif
                    if is_edt and pres in PRESENCE_PROFILES:
                        proba_h = PRESENCE_PROFILES[pres]
                        if proba_h:
                            st.markdown("**Courbe de présence au domicile (semaine)**")
                            df_pres = pd.DataFrame({
                                "Heure":     [f"{h:02d}h" for h in range(24)],
                                "% présent": [round(p*100, 1) for p in proba_h],
                            })
                            st.bar_chart(df_pres.set_index("Heure"),
                                         height=120, use_container_width=True)

    # ── Onglet Équipements ────────────────────────────────────────────────────
    with tab_equip:
        st.subheader("Gestion des équipements")
        members_now = st.session_state.get("members", [
            {"age_group": "Adulte actif (18-65 ans)", "absences_semaine": [(8, 17)]}
        ])
        n_persons = len(members_now)
        l_scale   = lights_factor(n_persons)
        catalogue_override = {}
        st.markdown("#### 🔌 Bruit de fond")
        st.caption("Inclut : box/router (~10W) · veilles TV/PC/HiFi (~15W) "
                   "· kettle/toaster/laptop (~15W) · chargeurs divers (~5W)")
        bruit_fond_W = st.slider(
            "Bruit de fond constant (W)",
            0, 150, 45, step=5,
            help="Valeur par défaut 45W = box 10W + veilles 20W + kettle/toaster/laptop 15W"
        )

        # ── Éclairage ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💡 Éclairage")
        st.caption(
            "L'éclairage est modélisé comme une puissance continue active "
            "pendant les périodes de présence, modulée par l'ensoleillement "
            "selon la saison. Plus il fait nuit, plus l'éclairage est fort."
        )
        lights_active = st.checkbox("Activer l'éclairage", value=True,
                                    key="lights_active")
        lights_w = st.slider("Puissance par membre présent (W)",
                             5, 100, int(LIGHTS_W_PER_MEMBER), step=5,
                             key="lights_w_per_member",
                             help="Puissance estimée par personne présente dans le logement")
        if "lights" not in catalogue_override:
            catalogue_override["lights"] = CATALOGUE.get("lights", {}).copy()
        catalogue_override["lights"].update({
            "active":            lights_active,
            "lights_w_per_member": float(lights_w),
        })
        st.markdown("---")
        st.markdown("#### 🧊 Réfrigérateur")

        if "fridge" not in catalogue_override:
            catalogue_override["fridge"] = CATALOGUE["fridge"].copy()
        fridge_cat = catalogue_override["fridge"]
        cycles_fridge = cycles_data.get("fridge", [])

        # Checkbox active/inactive
        fc0, fc1 = st.columns([1, 4])
        with fc0:
            fridge_active = st.checkbox(
                "Actif", value=fridge_cat.get("active", True), key="fridge_active"
            )
        source_fridge = CYCLE_SOURCE.get("fridge", "REFIT")
        badge_fridge  = "🟣 REFIT" if "REFIT" in source_fridge else "🔵 SmartHouse"
        with fc1:
            st.caption(
                f"{badge_fridge} · {len(cycles_fridge)} cycle(s) disponible(s) · "
                f"Continu — cycles enchaînés avec gap OFF entre chaque cycle"
            )
        fridge_cat["active"] = fridge_active

        if fridge_active:
            with st.expander("📏 Calibrage Réfrigérateur", expanded=False):
                all_durs_f  = sorted([len(c) for c in cycles_fridge]) if cycles_fridge else []
                all_peaks_f = [c.max() for c in cycles_fridge] if cycles_fridge else []

                if cycles_fridge:
                    st.caption(
                        f"{len(cycles_fridge)} cycle(s) · "
                        f"durées : {min(all_durs_f)}-{max(all_durs_f)} min · "
                        f"pics : {int(min(all_peaks_f))}-{int(max(all_peaks_f))} W"
                    )

                # Gap OFF
                st.markdown("**Gap OFF entre cycles**")
                fg1, fg2 = st.columns(2)
                with fg1:
                    gap_min_v = st.number_input(
                        "Min (min)", 10, 180,
                        int(fridge_cat.get("gap_off_min", 60)), step=5,
                        key="fridge_gap_min"
                    )
                    fridge_cat["gap_off_min"] = gap_min_v
                with fg2:
                    gap_max_v = st.number_input(
                        "Max (min)", 10, 240,
                        int(fridge_cat.get("gap_off_max", 90)), step=5,
                        key="fridge_gap_max"
                    )
                    fridge_cat["gap_off_max"] = gap_max_v

                # Filtre durée
                st.markdown("**Filtre durée**")
                fd1, fd2 = st.columns(2)
                d_lo = min(all_durs_f) if all_durs_f else 20
                d_hi = max(all_durs_f) if all_durs_f else 60
                with fd1:
                    dur_min_f = st.number_input(
                        "Min (min)", 1, 300, d_lo, step=5, key="fridge_dur_min"
                    )
                with fd2:
                    dur_max_f = st.number_input(
                        "Max (min)", 1, 300, d_hi, step=5, key="fridge_dur_max"
                    )
                fridge_cat["dur_filter"] = [int(dur_min_f), int(dur_max_f)]

                # Calibrage avancé
                st.markdown("**Calibrage avancé**")
                fa1, fa2, fa3 = st.columns(3)
                with fa1:
                    e_target_f = st.number_input(
                        "Energie cible/cycle (Wh)", 0.0, 500.0,
                        float(fridge_cat.get("energy_target_wh") or 0.0),
                        step=5.0, key="fridge_e_target",
                        help="Calibrage energetique : N repetitions du cycle. 0 = desactive."
                    )
                    if e_target_f > 0:
                        fridge_cat["energy_target_wh"] = float(e_target_f)
                        if cycles_fridge:
                            e_sim = float(np.median([c.sum()/60 for c in cycles_fridge]))
                            ratio_e = e_target_f / max(e_sim, 1)
                            st.caption(f"E sim : {e_sim:.0f}Wh → ratio x{ratio_e:.2f}")
                    else:
                        fridge_cat.pop("energy_target_wh", None)
                with fa2:
                    noise_f = st.number_input(
                        "Offset bruit (W)", 0.0, 100.0,
                        float(fridge_cat.get("power_noise_floor_w") or 0.0),
                        step=2.0, key="fridge_noise",
                        help="Soustrait ce plancher manuellement. 0 = calcul auto depuis puissance moyenne cible."
                    )
                    fridge_cat["power_noise_floor_w"] = float(noise_f) if noise_f > 0 else None
                with fa3:
                    annual_mean_f = st.number_input(
                        "Pmoy annuelle cible (W)", 0.0, 300.0,
                        float(fridge_cat.get("annual_mean_target_w") or 0.0),
                        step=1.0, key="fridge_annual_mean",
                        help="Puissance moyenne annuelle mesurée sur ce foyer. "
                             "Si > 0, l'offset bruit est calculé automatiquement "
                             "pour corriger l'écart simulé/réel. 0 = desactive."
                    )
                    if annual_mean_f > 0:
                        fridge_cat["annual_mean_target_w"] = float(annual_mean_f)
                        if cycles_fridge:
                            gap_avg = (fridge_cat.get("gap_off_min", 60) +
                                       fridge_cat.get("gap_off_max", 90)) / 2.0
                            sample  = cycles_fridge[0]
                            dur_c   = len(sample)
                            mean_sim = sample.sum() / (dur_c + gap_avg)
                            diff     = mean_sim - annual_mean_f
                            if diff > 0:
                                ratio_dil = (dur_c + gap_avg) / dur_c
                                offset    = diff * ratio_dil
                                st.caption(
                                    f"Pmoy sim : {mean_sim:.1f}W → "
                                    f"ecart : {diff:.1f}W → "
                                    f"offset concentre : {offset:.1f}W"
                                )
                            else:
                                st.caption(f"Pmoy sim : {mean_sim:.1f}W — pas de correction necessaire")
                    else:
                        fridge_cat.pop("annual_mean_target_w", None)

                # Aperçu effet offset
                if cycles_fridge and (noise_f > 0 or e_target_f > 0):
                    sample     = cycles_fridge[0].copy()
                    sample_cor = np.clip(sample - (noise_f or 0), 0, None)
                    st.caption(
                        f"Pic original : {sample.max():.0f}W → "
                        f"Pic corrige : {sample_cor.max():.0f}W  |  "
                        f"Moy active : {sample[sample>5].mean():.0f}W → "
                        f"{sample_cor[sample_cor>5].mean():.0f}W"
                        if sample_cor[sample_cor > 5].any() else
                        f"Pic original : {sample.max():.0f}W → {sample_cor.max():.0f}W"
                    )

        st.markdown("---")

        # ── Congélateur ───────────────────────────────────────────────────
        st.markdown("#### 🧊 Congélateur")

        if "freezer" not in catalogue_override:
            catalogue_override["freezer"] = CATALOGUE["freezer"].copy()
        freezer_cat = catalogue_override["freezer"]
        cycles_freezer = cycles_data.get("freezer", [])

        fz0, fz1 = st.columns([1, 4])
        with fz0:
            freezer_active = st.checkbox(
                "Actif", value=freezer_cat.get("active", True), key="freezer_active"
            )
        source_freezer = CYCLE_SOURCE.get("freezer", "LPG")
        badge_freezer  = "🟣 REFIT" if "REFIT" in source_freezer else (
            "🔵 SmartHouse" if "SmartHouse" in source_freezer else "🟠 LPG"
        )
        with fz1:
            st.caption(
                f"{badge_freezer} · {len(cycles_freezer)} cycle(s) disponible(s) · "
                f"Usage : continu (cycles enchaînés)"
            )
        freezer_cat["active"] = freezer_active

        if freezer_active:
            with st.expander("📏 Calibrage Congélateur", expanded=False):
                all_durs_fz  = sorted([len(c) for c in cycles_freezer]) if cycles_freezer else []
                all_peaks_fz = [c.max() for c in cycles_freezer] if cycles_freezer else []

                if cycles_freezer:
                    st.caption(
                        f"{len(cycles_freezer)} cycle(s) · "
                        f"Durées : {all_durs_fz[0]}-{all_durs_fz[-1]} min · "
                        f"Pics : {min(all_peaks_fz):.0f}-{max(all_peaks_fz):.0f} W"
                    )
                fzc1, fzc2, fzc3 = st.columns(3)
                with fzc1:
                    fz_gap_min = st.number_input(
                        "Gap OFF min (min)", 10, 120,
                        int(freezer_cat.get("gap_off_min", 60)), step=5,
                        key="freezer_gap_min"
                    )
                    freezer_cat["gap_off_min"] = fz_gap_min
                with fzc2:
                    fz_gap_max = st.number_input(
                        "Gap OFF max (min)", 10, 180,
                        int(freezer_cat.get("gap_off_max", 90)), step=5,
                        key="freezer_gap_max"
                    )
                    freezer_cat["gap_off_max"] = fz_gap_max
                with fzc3:
                    if all_durs_fz:
                        d_lo = int(freezer_cat.get("dur_filter", [all_durs_fz[0], all_durs_fz[-1]])[0])
                        d_hi = int(freezer_cat.get("dur_filter", [all_durs_fz[0], all_durs_fz[-1]])[1])
                    else:
                        d_lo, d_hi = 10, 60
                    fz_dur_min = st.number_input(
                        "Durée min (min)", 1, 300, d_lo, step=5, key="freezer_dur_min"
                    )
                    fz_dur_max = st.number_input(
                        "Durée max (min)", 1, 300, d_hi, step=5, key="freezer_dur_max"
                    )
                freezer_cat["dur_filter"] = [int(fz_dur_min), int(fz_dur_max)]

                fze1, fze2 = st.columns(2)
                with fze1:
                    fz_noise = st.number_input(
                        "Plancher bruit (W)", 0.0, 200.0,
                        float(freezer_cat.get("power_noise_floor_w") or 0.0),
                        step=2.0, key="freezer_noise",
                        help="Soustraction du plancher de bruit du compresseur"
                    )
                    freezer_cat["power_noise_floor_w"] = float(fz_noise) if fz_noise > 0 else None
                with fze2:
                    fz_annual = st.number_input(
                        "Cible moyenne annuelle (W)", 0.0, 200.0,
                        float(freezer_cat.get("annual_mean_target_w") or 0.0),
                        step=1.0, key="freezer_annual_mean",
                        help="Puissance moyenne sur l'année (issue des données mesurées)"
                    )
                    if fz_annual > 0:
                        freezer_cat["annual_mean_target_w"] = float(fz_annual)
                    else:
                        freezer_cat.pop("annual_mean_target_w", None)

                if cycles_freezer and fz_noise > 0:
                    sample_fz     = cycles_freezer[0].copy()
                    sample_fz_cor = np.clip(sample_fz - fz_noise, 0, None)
                    st.caption(
                        f"Pic original : {sample_fz.max():.0f}W → "
                        f"Pic corrigé : {sample_fz_cor.max():.0f}W  |  "
                        f"Moy active : {sample_fz[sample_fz>5].mean():.0f}W → "
                        f"{sample_fz_cor[sample_fz_cor>5].mean():.0f}W"
                        if sample_fz_cor[sample_fz_cor > 5].any() else
                        f"Pic original : {sample_fz.max():.0f}W → {sample_fz_cor.max():.0f}W"
                    )

        st.markdown("---")

        # ── Probas pondérées calculées (utilisées en interne) ───────────────
        members_now = st.session_state.get("members", [])
        weighted_probs = compute_weighted_probs(members_now)

        # Distribution horaire : déterminée par le profil de présence du membre
        # (Adulte actif → HOURLY_WEIGHTS_INSEE["Adulte actif"], etc.)
        # Pas de sélecteur global — chaque membre applique son propre profil EDT
        st.session_state["insee_profil"] = "Défaut catalogue"
        st.markdown("---")
        st.markdown("#### Probabilités et activation par appareil")
        # Regroupement visuel

        # ── Facteurs par âge modifiables ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 👤 Facteurs d'usage par membre")
        st.caption(
            "Facteurs multiplicatifs sur la probabilité de base de chaque usage. "
            "**1.0 = référence adulte actif.** "
            "Les valeurs suggérées ci-dessous sont pré-remplies d'après "
            "l'Enquête Emploi du Temps INSEE pour chaque tranche d'âge. "
            "Modifiables librement."
        )
        members_now = st.session_state.get("members", [])
        age_factors_override = {}
        FACTEURS_PAR_AGE_DEFAUT = {
            "Bébé (0-3 ans)": {
                "lights": 1.2, "tv": 0.8, "gaming": 0.0, "desktop_pc": 0.0,
                "washing_machine": 1.5, "dishwasher": 1.1, "vacuum": 1.4,
                "hair_dryer": 0.0, "coffee_machine": 1.0, "oven": 1.1, "hob": 1.1,
            },
            "Enfant (4-11 ans)": {
                "lights": 1.1, "tv": 1.1, "gaming": 1.8, "desktop_pc": 0.8,
                "washing_machine": 1.2, "dishwasher": 1.1, "vacuum": 1.2,
                "hair_dryer": 0.3, "coffee_machine": 1.0, "oven": 1.1, "hob": 1.1,
            },
            "Adolescent (12-17 ans)": {
                "lights": 1.0, "tv": 1.2, "gaming": 2.0, "desktop_pc": 1.5,
                "washing_machine": 1.1, "dishwasher": 1.0, "vacuum": 1.0,
                "hair_dryer": 1.3, "coffee_machine": 1.0, "oven": 1.0, "hob": 1.0,
            },
            "Adulte actif (18-65 ans)": {
                "lights": 1.0, "tv": 1.0, "gaming": 0.2, "desktop_pc": 1.0,
                "washing_machine": 1.0, "dishwasher": 1.0, "vacuum": 1.0,
                "hair_dryer": 1.0, "coffee_machine": 1.0, "oven": 1.0, "hob": 1.0,
            },
            "Senior (65+ ans)": {
                "lights": 1.2, "tv": 1.5, "gaming": 0.0, "desktop_pc": 0.3,
                "washing_machine": 0.9, "dishwasher": 1.0, "vacuum": 0.8,
                "hair_dryer": 0.7, "coffee_machine": 1.2, "oven": 1.1, "hob": 1.1,
            },
        }
        USAGES_FACTEUR = {
            "lights": "Éclairage", "tv": "TV", "gaming": "Console",
            "desktop_pc": "PC fixe", "washing_machine": "Lave-linge",
            "dishwasher": "Lave-vaisselle", "vacuum": "Aspirateur",
            "hair_dryer": "Sèche-cheveux", "coffee_machine": "Cafetière",
            "oven": "Four", "hob": "Plaques",
        }
        ages_presents = list({m.get("age_group", "Adulte actif (18-65 ans)")
                              for m in members_now})
        for age in ages_presents:
            edt_profil = AGE_TO_EEDT_PROFIL.get(age)
            has_edt    = edt_profil is not None and edt_profil in HOURLY_WEIGHTS_INSEE
            # Vérifie si le membre utilise le profil EDT ou Personnalisé
            membre_profil_pres = next(
                (m.get("presence_profil", "Personnalisé")
                 for m in members_now
                 if m.get("age_group") == age),
                "Personnalisé"
            )
            mode_edt = has_edt and membre_profil_pres != "Personnalisé"
            with st.expander(f"👤 {age} — {'✅ EDT' if mode_edt else '⚙️ Personnalisé'}", expanded=False):
                defauts = FACTEURS_PAR_AGE_DEFAUT.get(age, {})
                factors = {}
                if mode_edt:
                    st.caption(
                        f"Profil **{edt_profil}** (Enquête Emploi du Temps INSEE) — "
                        f"distribution horaire observée. Modifiable heure par heure ci-dessous."
                    )
                    edt_weights  = HOURLY_WEIGHTS_INSEE[edt_profil]
                    hours_labels = [f"{h:02d}h" for h in range(24)]
                    edt_usages   = [u for u in USAGES_FACTEUR if u in edt_weights]
                    if edt_usages:
                        st.markdown("**Distribution horaire (semaine) :**")
                        for usage in edt_usages:
                            w_default = edt_weights[usage]
                            lbl = USAGES_FACTEUR[usage]
                            st.caption(f"⏱ {lbl}")
                            with st.container():
                                df_w = pd.DataFrame({
                                    "h": hours_labels,
                                    "%": [round(x*100, 1) for x in w_default]
                                })
                                st.bar_chart(df_w.set_index("h"),
                                             height=70, use_container_width=True)
                                st.caption("Modifiez les poids ci-dessous (normalisés automatiquement) :")
                                cols_w = st.columns(8)
                                new_weights = []
                                for h in range(24):
                                    with cols_w[h % 8]:
                                        val = st.number_input(
                                            f"{h:02d}h", 0.0, 1.0,
                                            float(round(w_default[h], 3)),
                                            step=0.01,
                                            key=f"edt_w_{age}_{usage}_{h}",
                                            format="%.2f",
                                            label_visibility="visible"
                                        )
                                        new_weights.append(val)
                                total = sum(new_weights) or 1.0
                                norm_w = [v / total for v in new_weights]
                                if usage in catalogue_override:
                                    catalogue_override[usage]["hourly_weights_semaine"] = norm_w
                                    catalogue_override[usage]["hourly_weights_vacances"] = norm_w
                            factors[usage] = 1.0
                    non_edt = [u for u in USAGES_FACTEUR if u not in edt_weights]
                    if non_edt:
                        st.markdown("**Usages non couverts par l'EDT :**")
                        cols_ne = st.columns(4)
                        for idx, usage in enumerate(non_edt):
                            default_val = float(defauts.get(usage, 1.0))
                            with cols_ne[idx % 4]:
                                factors[usage] = st.number_input(
                                    USAGES_FACTEUR[usage], 0.0, 3.0, default_val,
                                    step=0.1, key=f"af_{age}_{usage}", format="%.1f"
                                )
                else:
                    # Mode Personnalisé → facteurs multiplicatifs
                    st.caption(
                        "Mode personnalisé — facteurs multiplicatifs sur la probabilité "
                        "de base. **1.0 = référence adulte actif.**"
                    )
                    cols = st.columns(4)
                    for idx, (usage, label) in enumerate(USAGES_FACTEUR.items()):
                        default_val = float(defauts.get(usage, 1.0))
                        with cols[idx % 4]:
                            factors[usage] = st.number_input(
                                label, 0.0, 3.0, default_val,
                                step=0.1, key=f"af_{age}_{usage}", format="%.1f"
                            )
                age_factors_override[age] = factors
        st.session_state["age_factors_override"] = age_factors_override
        st.markdown("---")
        groupes_affichage = {
            "🍳 Cuisine": ["hob", "oven", "microwave", "bread_maker",
                           "extractor_hood", "small_cooking",
                           "kettle", "coffee_machine", "toaster"],
            "🧺 Électroménager": ["washing_machine", "dishwasher", "dryer",
                                   "vacuum", "hair_dryer"],
            "📺 Divertissement": ["tv", "sat_hifi", "gaming", "beamer"],
            "💻 Informatique": ["desktop_pc", "laptop", "printer"],
            "💡 Autres": ["lights", "fridge", "air_conditioner",
                          "garden"],
        }
        for groupe_label, usages in groupes_affichage.items():
            with st.expander(groupe_label, expanded=False):
                for usage in usages:
                    if usage not in CATALOGUE:
                        continue
                    if usage in ("fridge", "freezer"):
                        continue  # gérés séparément avec section dédiée
                    cat  = CATALOGUE[usage].copy()
                    disp = cat["label"]
                    cycles_raw   = cycles_data.get(usage, [])
                    cycles_avail = len(cycles_raw)
                    source_str   = CYCLE_SOURCE.get(usage, "LPG")
                    # Profil distribution horaire INSEE global
                    cat["insee_profil"] = st.session_state.get(
                        "insee_profil", "Défaut catalogue")
                    # Probas EDT pondérées selon composition foyer
                    members_now2 = st.session_state.get("members", [])
                    if usage in weighted_probs and members_now2:
                        has_edt_member = any(
                            m.get("presence_profil", "Personnalisé") != "Personnalisé"
                            for m in members_now2
                        )
                        if has_edt_member:
                            cat["prob_semaine"]      = weighted_probs[usage]
                            cat["force_no_binomial"] = True
                    source_badge = (
                        "🔵 SmartHouse" if "SmartHouse" in source_str else
                        "🟣 REFIT"      if "REFIT"      in source_str else
                        "🟠 LPG"
                    )

                    # ── Statut liaison dynamique ──────────────────────────────
                    liaisons_now = st.session_state.get("liaisons", LIAISONS_DEFAULT)
                    is_source = [l for l in liaisons_now
                                 if l["source"] == usage and l.get("active", True)]
                    is_cible = []
                    for l in liaisons_now:
                        if l["cible"] == usage and l.get("active", True):
                            # Cherche dans catalogue_override, sinon dans CATALOGUE
                            # La source peut ne pas être encore dans catalogue_override
                            # au premier render — on se rabat sur CATALOGUE
                            src_cat = catalogue_override.get(
                                l["source"],
                                CATALOGUE.get(l["source"], {})
                            )
                            # Active par défaut si pas encore dans override
                            src_active = src_cat.get("active", True)
                            prob_s = src_cat.get("prob_semaine",
                                CATALOGUE.get(l["source"], {}).get("prob_semaine", 0))
                            prob_w = src_cat.get("prob_weekend",
                                CATALOGUE.get(l["source"], {}).get("prob_weekend", 0))
                            if src_active and (prob_s > 0 or prob_w > 0):
                                is_cible.append(l)
                    if is_cible:
                        src_labels = ", ".join(
                            CATALOGUE.get(l["source"], {}).get("label", l["source"])
                            for l in is_cible
                        )
                        liaison_badge = f"🔗 Lié (déclenché par : {src_labels})"
                        is_linked = True
                    elif is_source:
                        cible_labels = ", ".join(
                            CATALOGUE.get(l["cible"], {}).get("label", l["cible"])
                            for l in is_source
                        )
                        liaison_badge = f"🔗 Lié (déclenche : {cible_labels})"
                        is_linked = False  # source = toujours configurable
                    else:
                        liaison_badge = ""
                        is_linked = False

                    # ── Layout principal ──────────────────────────────────────
                    if is_cible:
                        # Appareil cible : checkbox + badge liaison + config toujours visible
                        c1, c2 = st.columns([3, 5])
                        with c1:
                            active = st.checkbox(
                                disp, value=cat.get("active", True), key=f"active_{usage}"
                            )
                            st.caption(
                                f"{source_badge} · {cycles_avail} cycle(s)  \n"
                                f"{liaison_badge}"
                            )
                        cat.update({"active": active})
                        # Config prob si liaison désactivée
                        with c2:
                            st.caption(
                                "⚙️ Si la liaison est **active**, la probabilité "
                                "individuelle est ignorée. Si la liaison est **supprimée**, "
                                "configurer ci-dessous."
                            )
                            prob_s = st.slider("Semaine (si non lié)", 0.0, 1.0,
                                               cat["prob_semaine"], step=0.05,
                                               key=f"prob_s_{usage}",
                                               disabled=len(is_cible) > 0)
                            prob_w = st.slider("Weekend (si non lié)", 0.0, 1.0,
                                               cat["prob_weekend"], step=0.05,
                                               key=f"prob_w_{usage}",
                                               disabled=len(is_cible) > 0)
                        cat.update({"prob_semaine": prob_s, "prob_weekend": prob_w})
                    else:
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            active = st.checkbox(
                                disp, value=True, key=f"active_{usage}"
                            )
                            caption_txt = f"{source_badge} · {cycles_avail} cycle(s)"
                            if liaison_badge:
                                caption_txt += f"  \n{liaison_badge}"
                            st.caption(caption_txt)
                        with c2:
                            prob_s = st.slider("Semaine", 0.0, 1.0,
                                               cat["prob_semaine"], step=0.05,
                                               key=f"prob_s_{usage}")
                        with c3:
                            prob_w = st.slider("Weekend", 0.0, 1.0,
                                               cat["prob_weekend"], step=0.05,
                                               key=f"prob_w_{usage}")
                        immo_default = IMMOBILISATION_DEFAULT.get(usage, 0.0)
                        immo_val = st.slider(
                            f"Immobilisation membre ({usage})",
                            0.0, 1.0, immo_default, step=0.1,
                            key=f"immo_{usage}",
                            help="0 = usage passif · 1 = usage actif exclusif",
                            label_visibility="visible"
                        )
                        cat["immobilisation"] = immo_val
                        cat.update({"prob_semaine": prob_s, "prob_weekend": prob_w})
                    cat["active"] = active

                    # ── Paramètres spécifiques au frigo ──────────────────────
                    if usage == "fridge":
                        st.caption("🌡️ Calibrage Réfrigérateur")
                        with st.container():
                            fc1, fc2, fc3 = st.columns(3)
                            with fc1:
                                gap_min_v = st.number_input(
                                    "Gap OFF min (min)", 10, 120,
                                    int(cat.get("gap_off_min", 60)), step=5,
                                    key="fridge_gap_min"
                                )
                                cat["gap_off_min"] = gap_min_v
                            with fc2:
                                gap_max_v = st.number_input(
                                    "Gap OFF max (min)", 10, 180,
                                    int(cat.get("gap_off_max", 90)), step=5,
                                    key="fridge_gap_max"
                                )
                                cat["gap_off_max"] = gap_max_v
                            with fc3:
                                offset_v = st.number_input(
                                    "Offset puissance (W)", -100, 200,
                                    int(cat.get("power_offset_w", 0)), step=5,
                                    key="fridge_offset",
                                    help="Soustrait cette valeur au cycle simulé "
                                         "pour recalibrer la puissance sur le foyer réel. "
                                         "Positif = réduit la puissance."
                                )
                                cat["power_offset_w"] = float(offset_v)
                            # Aperçu de l'effet de l'offset
                            if cycles_raw and offset_v != 0:
                                sample = cycles_raw[0].copy()
                                sample_corr = np.clip(sample - offset_v, 0, None)
                                st.caption(
                                    f"Pic original : {sample.max():.0f}W → "
                                    f"Pic corrigé : {sample_corr.max():.0f}W  |  "
                                    f"Moy originale : {sample[sample>0].mean():.0f}W → "
                                    f"Moy corrigée : {sample_corr[sample_corr>0].mean():.0f}W"
                                    if sample_corr.any() else
                                    f"Pic original : {sample.max():.0f}W → "
                                    f"Pic corrigé : {sample_corr.max():.0f}W"
                                )

                    # ── Calibrage des cycles (option 3 + 4) ───────────────────
                    dur_def = CYCLE_DURATION_FILTER.get(usage)
                    all_durs = sorted([len(c) for c in cycles_raw]) if cycles_raw else []
                    _calib_lbl = "📏 Calibrage des cycles" + (f" · filtre [{dur_def[0]}-{dur_def[1]}min]" if dur_def else "")
                    st.caption(f"── {_calib_lbl}")
                    with st.container():
                        if not cycles_raw:
                            st.warning("Aucun cycle disponible pour cet usage.")
                        else:
                            # Stats des cycles disponibles
                            all_peaks = [c.max() for c in cycles_raw]
                            st.caption(
                                f"{cycles_avail} cycle(s) · "
                                f"durées : {min(all_durs)}-{max(all_durs)} min · "
                                f"pics : {int(min(all_peaks))}-{int(max(all_peaks))} W"
                            )
                            # Filtre durée
                            st.markdown("**Durée**")
                            dc1, dc2 = st.columns(2)
                            d_lo = dur_def[0] if dur_def else min(all_durs)
                            d_hi = dur_def[1] if dur_def else max(all_durs)
                            with dc1:
                                dur_min = st.number_input(
                                    "Min (min)", 1, 600, d_lo, step=5,
                                    key=f"dur_min_{usage}"
                                )
                            with dc2:
                                dur_max = st.number_input(
                                    "Max (min)", 1, 600, d_hi, step=5,
                                    key=f"dur_max_{usage}"
                                )
                            cat["dur_filter"] = [int(dur_min), int(dur_max)]
                            # Filtre puissance
                            st.markdown("**Puissance de pic**")
                            pk1, pk2 = st.columns(2)
                            with pk1:
                                peak_min = st.number_input(
                                    "Min (W)", 0, 15000,
                                    int(min(all_peaks)),
                                    step=50, key=f"peak_min_{usage}"
                                )
                            with pk2:
                                peak_max = st.number_input(
                                    "Max (W)", 0, 15000,
                                    int(max(all_peaks)),
                                    step=50, key=f"peak_max_{usage}"
                                )
                            if peak_min > 0 or peak_max < int(max(all_peaks)):
                                cat["peak_filter"] = [int(peak_min), int(peak_max)]
                            # Cycles dans la plage durée + puissance (option 4)
                            in_range = [c for c in cycles_raw
                                        if dur_min <= len(c) <= dur_max
                                        and peak_min <= c.max() <= peak_max]
                            n_in = len(in_range)
                            if n_in > 0:
                                st.success(
                                    f"✅ {n_in} cycle(s) réel(s) dans la plage "
                                    f"[{int(dur_min)}-{int(dur_max)} min · "
                                    f"{int(peak_min)}-{int(peak_max)} W]"
                                )
                            else:
                                # Aucun cycle dans la plage filtre — affichage
                                # du cycle réel le plus proche, sans interpolation
                                # (la fabrication par étirement a été supprimée).
                                target = int((dur_min + dur_max) / 2)
                                best   = min(cycles_raw, key=lambda c: abs(len(c) - target))
                                st.info(
                                    f"Aucun cycle dans [{int(dur_min)}-{int(dur_max)}min] · "
                                    f"cycle réel le plus proche affiché : {len(best)} min "
                                    f"(utilisé tel quel, sans étirement)"
                                )
                                fig_c, ax_c = plt.subplots(figsize=(6, 2))
                                ax_c.plot(best, color="#FF9800", alpha=0.9,
                                          linewidth=1.2, label=f"Cycle réel ({len(best)}min)")
                                ax_c.set_xlabel("Minute")
                                ax_c.set_ylabel("W")
                                ax_c.legend(fontsize=7)
                                ax_c.grid(alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig_c, use_container_width=True)
                                plt.close(fig_c)

                            # Ajustement énergétique (ratio mesuré/cycle) + offset bruit
                            st.markdown("**Calibrage avancé**")
                            adv1, adv2 = st.columns(2)
                            with adv1:
                                e_target = st.number_input(
                                    "Energie cible/cycle (Wh)",
                                    min_value=0.0,
                                    value=float(cat.get("energy_target_wh") or 0.0),
                                    step=10.0,
                                    key=f"e_target_{usage}",
                                    help="Répète le cycle un nombre entier de fois pour "
                                         "atteindre l'énergie mesurée (valide si écart "
                                         "énergie <20% et durée cible <30%)."
                                )
                                if e_target > 0:
                                    cat["energy_target_wh"] = float(e_target)
                                    if cycles_raw:
                                        e_sim = float(np.median([c.sum()/60 for c in cycles_raw]))
                                        dur_med = float(np.median(all_durs))
                                        
                                        # Récupère la cible temporelle
                                        dur_cible = (cat.get("dur_filter", [dur_med, dur_med])[0] + cat.get("dur_filter", [dur_med, dur_med])[1]) / 2.0
                                        
                                        # Calcul de la meilleure répétition théorique pour l'UI
                                        N = max(1, round(e_target / max(e_sim, 1)))
                                        e_new = e_sim * N
                                        dur_new = dur_med * N
                                        
                                        e_ok = abs(e_new - e_target) <= 0.20 * e_target
                                        dur_ok = abs(dur_new - dur_cible) <= 0.30 * dur_cible
                                        status = "✅ Validé" if (e_ok and dur_ok) else "❌ Rejeté"
                                        
                                        st.caption(
                                            f"Base : {e_sim:.0f}Wh ({dur_med:.0f}min) → "
                                            f"Répétition x{N} → {e_new:.0f}Wh ({dur_new:.0f}min) | {status}"
                                        )
                            with adv2:
                                noise_v = st.number_input(
                                    "Offset bruit (W)",
                                    min_value=0.0,
                                    value=float(cat.get("power_noise_floor_w") or 0.0),
                                    step=5.0,
                                    key=f"noise_{usage}",
                                    help="Soustrait ce plancher de bruit a tous les "
                                         "cycles. 0 = calcul automatique (P10 actif)."
                                )
                                cat["power_noise_floor_w"] = float(noise_v) if noise_v > 0 else None
                        # Distribution horaire — expander compact
                        has_density = usage in HOURLY_WEIGHTS_SEMAINE
                        st.caption(f"⏱ Distribution horaire {'(densité)' if has_density else '(gaussienne)'}")
                        with st.container():
                            if has_density:
                                default_w  = HOURLY_WEIGHTS_SEMAINE[usage]
                                default_we = HOURLY_WEIGHTS_WEEKEND.get(usage, default_w)
                                st.caption(f"Source : {source_str} · Modifiez les 24 valeurs (virgule-séparées)")
                                # 1 text_input au lieu de 24 number_input → 24× moins de widgets
                                hw_key = f"hw_txt_{usage}"
                                if hw_key not in st.session_state:
                                    st.session_state[hw_key] = ", ".join(
                                        str(round(v, 3)) for v in default_w)
                                hw_txt = st.text_input(
                                    "Poids 0h→23h", key=hw_key,
                                    label_visibility="visible"
                                )
                                try:
                                    new_weights = [float(x.strip())
                                                   for x in hw_txt.split(",")]
                                    if len(new_weights) != 24:
                                        raise ValueError
                                except Exception:
                                    new_weights = list(default_w)
                                    st.caption("⚠️ 24 valeurs attendues.")
                                total = sum(new_weights) or 1.0
                                normalized = [w / total for w in new_weights]
                                df_hw = pd.DataFrame({
                                    "Heure": [f"{h:02d}h" for h in range(24)],
                                    "Poids": normalized,
                                })
                                st.bar_chart(df_hw.set_index("Heure"),
                                             height=80, use_container_width=True)
                                cat[f"hourly_weights_semaine"] = normalized
                                cat[f"hourly_weights_weekend"] = [
                                    WEEKEND_ALPHA * n + (1 - WEEKEND_ALPHA) / 24
                                    for n in normalized
                                ]
                            else:
                                # Fallback gaussienne
                                h_pic = st.number_input(
                                    "Heure pic", 0.0, 23.0,
                                    float(cat.get("heure_pic", 12)),
                                    step=0.5, key=f"hpic_{usage}"
                                )
                                cat.update({
                                    "heure_pic":     h_pic,
                                    "semaine_h_pic": h_pic,
                                    "weekend_h_pic": h_pic,
                                })
                        cat.update({
                            "active":       active,
                            "prob_semaine": prob_s,
                            "prob_weekend": prob_w,
                        })
                    catalogue_override[usage] = cat
        st.session_state["catalogue_override"] = catalogue_override
        st.session_state["bruit_fond_W"]       = bruit_fond_W
        st.session_state["lights_scale"]       = l_scale
        # Forçage manuel
        st.markdown("---")
        st.markdown("---")
        st.subheader("⚡ Événements programmés (Forcés)")
        st.caption(
            "Ces événements s'exécutent **indépendamment de la présence** ou du sommeil "
            "des occupants. Idéal pour modéliser des appareils qui tournent en "
            "Heures Creuses la nuit (Lave-linge) ou pendant les absences (Aspirateur robot)."
        )
        if "forced_events" not in st.session_state:
            st.session_state["forced_events"] = []

        with st.container():
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                forced_usage = st.selectbox(
                    "Appareil à programmer", options=list(CATALOGUE.keys()),
                    format_func=lambda k: CATALOGUE[k]["label"],
                    key="forced_usage_sel"
                )
                forced_dtype = st.multiselect(
                    "Jours d'application",
                    ["semaine", "weekend", "vacances"],
                    default=["semaine", "weekend"],
                    key="forced_dtype_sel"
                )
            with f_col2:
                forced_mode = st.radio(
                    "Mode de déclenchement",
                    ["Distribution horaire réglable", "Heure fixe"],
                    horizontal=True, key="forced_mode_sel"
                )
                if forced_mode == "Heure fixe":
                    forced_heure = st.number_input(
                        "Heure exacte de démarrage",
                        min_value=0.0, max_value=23.5, value=3.0,
                        step=0.5, key="forced_heure_fixe"
                    )
                    forced_prob = st.slider(
                        "Probabilité (par jour)", 0.0, 1.0, 1.0,
                        step=0.05, key="forced_prob_fixe"
                    )
                    forced_w = None
                else:
                    forced_prob = st.slider(
                        "Probabilité (par jour)", 0.0, 1.0, 1.0,
                        step=0.05, key="forced_prob_plage"
                    )
                    st.write("**Configuration de la distribution horaire**")
                    template = st.selectbox(
                        "Pré-configuration (Template)",
                        ["Uniforme (Toute la journée)",
                         "Heures Creuses de nuit (22h-6h)",
                         "Heures Creuses méridiennes (12h-14h + 2h-8h)",
                         "Personnalisé"],
                        key="forced_template_sel"
                    )
                    w_init = np.zeros(24)
                    if template == "Uniforme (Toute la journée)":
                        w_init[:] = 1.0
                    elif template == "Heures Creuses de nuit (22h-6h)":
                        w_init[22:24] = 1.0; w_init[0:6] = 1.0
                    elif template == "Heures Creuses méridiennes (12h-14h + 2h-8h)":
                        w_init[12:14] = 1.0; w_init[2:8] = 1.0
                    else:
                        w_init[:] = 0.1
                    if st.session_state.get("_prev_template") != template:
                        st.session_state["_prev_template"] = template
                        for h in range(24):
                            st.session_state[f"fw_input_{h}"] = float(w_init[h])
                    st.caption("✏️ Poids horaires (0h - 23h)")
                    with st.container():
                        cols_fw = st.columns(8)
                        updated_w = []
                        for h in range(24):
                            with cols_fw[h % 8]:
                                if f"fw_input_{h}" not in st.session_state:
                                    st.session_state[f"fw_input_{h}"] = float(w_init[h])
                                val = st.number_input(
                                    f"{h:02d}h", min_value=0.0, max_value=10.0,
                                    step=0.1, key=f"fw_input_{h}"
                                )
                                updated_w.append(val)
                        df_preview = pd.DataFrame({
                            "Heure": [f"{h:02d}h" for h in range(24)],
                            "Poids": updated_w
                        })
                        st.bar_chart(df_preview.set_index("Heure"),
                                     height=90, use_container_width=True)
                    total_w = sum(updated_w)
                    forced_w = [v / total_w for v in updated_w] if total_w > 0 \
                               else (np.ones(24) / 24).tolist()
                    forced_heure = None

            if st.button("➕ Ajouter ce programme", type="primary"):
                ev = {"usage": forced_usage, "types": forced_dtype, "prob": forced_prob}
                if forced_mode == "Heure fixe":
                    ev["mode"] = "fixed"
                    ev["heure"] = forced_heure
                    ev["desc"] = f"Heure fixe {forced_heure:.1f}h"
                else:
                    ev["mode"] = "programme"
                    ev["hourly_weights"] = forced_w
                    active_h = [h for h, v in enumerate(updated_w) if v > 0]
                    ev["desc"] = (
                        "Distribution 24h/24" if len(active_h) == 24
                        else f"Heures : {', '.join([f'{h}h' for h in active_h])}" if active_h
                        else "Uniforme"
                    )
                st.session_state["forced_events"].append(ev)

        # ── Affichage de la liste des programmes ─────────────────────────
        if st.session_state["forced_events"]:
            st.markdown("**Programmes actifs :**")
            to_remove = []
            for idx, ev in enumerate(st.session_state["forced_events"]):
                c1, c2 = st.columns([5, 1])
                with c1:
                    lbl   = CATALOGUE[ev["usage"]]["label"]
                    types = ", ".join(ev["types"])
                    prob_txt = f"Prob: {ev.get('prob', 1.0)*100:.0f}%"
                    desc = ev.get("desc", "")
                    if ev.get("mode") == "programme":
                        st.info(f"⏱️ **{lbl}** : {desc} | {prob_txt} | [{types}]")
                    else:
                        st.success(f"📌 **{lbl}** : {desc} | {prob_txt} | [{types}]")
                with c2:
                    if st.button("🗑️", key=f"del_ev_{idx}"):
                        to_remove.append(idx)
            for idx in reversed(to_remove):
                st.session_state["forced_events"].pop(idx)
            if to_remove:
                st.rerun()
                
# ── Onglet Fréquences ──────────────────────────────────────────
    with tab_freq:
        st.subheader("Fréquence et Répétition des usages")
        st.caption(
            "Définissez le nombre maximum d'utilisations par jour d'un appareil. "
            "Pour chaque utilisation supplémentaire, définissez la probabilité conditionnelle "
            "qu'elle se produise (sachant que la précédente a eu lieu). L'empêchement de "
            "superposition des cycles est garanti par le moteur physique."
        )
        
        # On garde juste la probabilité par défaut pour les cycles suivants
        DEFAULT_P2 = 0.5
        
        for groupe_label, usages in groupes_affichage.items():
            with st.expander(groupe_label, expanded=True):
                for usage in usages:
                    if usage not in CATALOGUE or usage in ["fridge", "lights"]:
                        continue
                        
                    cat_ov = catalogue_override.get(usage, {})
                    if not cat_ov.get("active", True):
                        continue
                        
                    label = CATALOGUE[usage].get("label", usage)
                    st.markdown(f"**{label}**")
                    
                    # -- LIGNE SEMAINE --
                    c1_s, c2_s = st.columns([2, 5])
                    with c1_s:
                        max_u_s = st.number_input(
                            f"Max/j (Semaine)", 1, 10,
                            int(cat_ov.get("max_uses_semaine", cat_ov.get("max_uses", 1))), 
                            key=f"max_uses_s_{usage}"
                        )
                        catalogue_override[usage]["max_uses_semaine"] = max_u_s
                    with c2_s:
                        if max_u_s > 1:
                            cols_s = st.columns(max_u_s - 1)
                            decay_s = []
                            old_decay_s = cat_ov.get("decay_probs_semaine", cat_ov.get("decay_probs", []))
                            for i in range(2, max_u_s + 1):
                                def_p = old_decay_s[i-2] if (i-2) < len(old_decay_s) else DEFAULT_P2
                                with cols_s[(i-2) % len(cols_s)]:
                                    p = st.number_input(
                                        f"Prob {i}ème (Sem)", 0.0, 1.0, float(def_p), step=0.05,
                                        key=f"decay_s_{usage}_{i}"
                                    )
                                    decay_s.append(p)
                            catalogue_override[usage]["decay_probs_semaine"] = decay_s
                        else:
                            st.caption("1 seule util. Semaine max.")
                            catalogue_override[usage]["decay_probs_semaine"] = []
                            
                    # -- LIGNE WEEKEND --
                    c1_w, c2_w = st.columns([2, 5])
                    with c1_w:
                        max_u_w = st.number_input(
                            f"Max/j (Weekend)", 1, 10,
                            int(cat_ov.get("max_uses_weekend", cat_ov.get("max_uses", 1))), 
                            key=f"max_uses_w_{usage}"
                        )
                        catalogue_override[usage]["max_uses_weekend"] = max_u_w
                    with c2_w:
                        if max_u_w > 1:
                            cols_w = st.columns(max_u_w - 1)
                            decay_w = []
                            old_decay_w = cat_ov.get("decay_probs_weekend", cat_ov.get("decay_probs", []))
                            for i in range(2, max_u_w + 1):
                                def_p = old_decay_w[i-2] if (i-2) < len(old_decay_w) else DEFAULT_P2
                                with cols_w[(i-2) % len(cols_w)]:
                                    p = st.number_input(
                                        f"Prob {i}ème (We)", 0.0, 1.0, float(def_p), step=0.05,
                                        key=f"decay_w_{usage}_{i}"
                                    )
                                    decay_w.append(p)
                            catalogue_override[usage]["decay_probs_weekend"] = decay_w
                        else:
                            st.caption("1 seule util. Weekend max.")
                            catalogue_override[usage]["decay_probs_weekend"] = []
                            
                    st.markdown("<hr style='margin: 0.5em 0; opacity: 0.2'>", unsafe_allow_html=True)
        st.session_state["catalogue_override"] = catalogue_override
    # ── Onglet Liaisons ───────────────────────────────────────────────────────
    with tab_liaisons:
        st.markdown("""
Les **liaisons logiques** définissent les enchaînements automatiques entre appareils.
Quand un appareil source se déclenche, les appareils cibles sont activés selon
le délai et la probabilité configurés.
""")
        # Initialise les liaisons en session si absentes
        if "liaisons" not in st.session_state:
            st.session_state["liaisons"] = [l.copy() for l in LIAISONS_DEFAULT]
        liaisons = st.session_state["liaisons"]

        # Labels lisibles pour les usages
        usage_labels = {k: v["label"] for k, v in CATALOGUE.items()}
        usage_options = list(usage_labels.keys())

        st.markdown("---")

        # ── Liaisons existantes ───────────────────────────────────────────────
        to_delete = []
        for idx, liaison in enumerate(liaisons):
            src_lbl   = usage_labels.get(liaison["source"], liaison["source"])
            cible_lbl = usage_labels.get(liaison["cible"],  liaison["cible"])
            active    = liaison.get("active", True)
            badge     = "✅" if active else "⬜"
            with st.expander(
                f"{badge} {src_lbl} → {cible_lbl}  "
                f"(délai {liaison['delai_min']}-{liaison['delai_max']}min · "
                f"prob {liaison.get('prob',1.0)*100:.0f}%)",
                expanded=False
            ):
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
                with c1:
                    new_src = st.selectbox(
                        "Source", usage_options,
                        index=usage_options.index(liaison["source"])
                              if liaison["source"] in usage_options else 0,
                        key=f"liais_src_{idx}",
                        format_func=lambda k: usage_labels.get(k, k)
                    )
                    liaison["source"] = new_src
                with c2:
                    new_cible = st.selectbox(
                        "Cible", usage_options,
                        index=usage_options.index(liaison["cible"])
                              if liaison["cible"] in usage_options else 0,
                        key=f"liais_cible_{idx}",
                        format_func=lambda k: usage_labels.get(k, k)
                    )
                    liaison["cible"] = new_cible
                with c3:
                    liaison["delai_min"] = st.number_input(
                        "Délai min (min)", -120, 480,
                        int(liaison.get("delai_min", 0)),
                        step=5, key=f"liais_dmin_{idx}",
                        help="Négatif = cible démarre AVANT la fin de la source"
                    )
                with c4:
                    liaison["delai_max"] = st.number_input(
                        "Délai max (min)", -120, 480,
                        int(liaison.get("delai_max", 0)),
                        step=5, key=f"liais_dmax_{idx}"
                    )
                    liaison["prob"] = st.number_input(
                        "Probabilité", 0.0, 1.0,
                        float(liaison.get("prob", 1.0)),
                        step=0.05, key=f"liais_prob_{idx}", format="%.2f"
                    )
                with c5:
                    liaison["attend_presence"] = st.checkbox(
                        "Attend présence",
                        value=liaison.get("attend_presence", False),
                        key=f"liais_pres_{idx}",
                        help="Si coché, attend qu'un membre soit présent avant de déclencher"
                    )
                    liaison["active"] = st.checkbox(
                        "Active",
                        value=liaison.get("active", True),
                        key=f"liais_act_{idx}"
                    )
                    if st.button("🗑 Supprimer", key=f"liais_del_{idx}",
                                 type="secondary"):
                        to_delete.append(idx)

                if liaison.get("note"):
                    st.caption(f"Note : {liaison['note']}")

        # Suppression après la boucle
        for idx in sorted(to_delete, reverse=True):
            liaisons.pop(idx)
        if to_delete:
            st.rerun()

        st.markdown("---")

        # ── Ajout d'une nouvelle liaison ─────────────────────────────────────
        st.markdown("#### ➕ Ajouter une liaison")
        na1, na2, na3, na4, na5 = st.columns([2, 2, 1, 1, 1])
        with na1:
            new_src = st.selectbox(
                "Source", usage_options, key="new_liais_src",
                format_func=lambda k: usage_labels.get(k, k)
            )
        with na2:
            new_cible = st.selectbox(
                "Cible", usage_options, key="new_liais_cible",
                format_func=lambda k: usage_labels.get(k, k)
            )
        with na3:
            new_dmin = st.number_input("Délai min", -120, 480, 0, step=5,
                                        key="new_liais_dmin")
            new_dmax = st.number_input("Délai max", -120, 480, 30, step=5,
                                        key="new_liais_dmax")
        with na4:
            new_prob = st.number_input("Probabilité", 0.0, 1.0, 1.0,
                                        step=0.05, key="new_liais_prob",
                                        format="%.2f")
        with na5:
            new_pres = st.checkbox("Attend présence", key="new_liais_pres")
            new_note = st.text_input("Note (optionnel)", key="new_liais_note")

        if st.button("➕ Ajouter cette liaison", type="primary"):
            liaisons.append({
                "source":          new_src,
                "cible":           new_cible,
                "delai_min":       int(new_dmin),
                "delai_max":       int(new_dmax),
                "prob":            float(new_prob),
                "attend_presence": new_pres,
                "active":          True,
                "note":            new_note,
            })
            st.success(f"Liaison ajoutée : {usage_labels.get(new_src, new_src)} → "
                       f"{usage_labels.get(new_cible, new_cible)}")
            st.rerun()

        # ── Reset par défaut ──────────────────────────────────────────────────
        if st.button("↺ Réinitialiser aux liaisons par défaut"):
            st.session_state["liaisons"] = [l.copy() for l in LIAISONS_DEFAULT]
            st.rerun()

    # ── Onglet Simulation ─────────────────────────────────────────────────────
    with tab_sim:
        st.subheader("Paramètres de simulation")
        # Sélecteur météo
        st.markdown("#### 🌡️ Profil météo")
        use_meteo = st.toggle("Utiliser les données météo réelles (Rennes-St Jacques)",
                              value=True, key="use_meteo")
        meteo_data_loaded = {}
        if use_meteo and not METEO_CSV.empty:
            avail_years = get_available_years(METEO_CSV, METEO_STATION)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                meteo_year = st.selectbox(
                    "Année météo de référence",
                    options=avail_years,
                    index=avail_years.index(2023) if 2023 in avail_years
                           else len(avail_years) - 1,
                    key="meteo_year",
                    help="Les données météo de cette année seront utilisées "
                         "pour moduler les probabilités d'usage."
                )
            with col_m2:
                st.metric("Station", METEO_STATION)
            meteo_data_loaded = load_meteo(METEO_CSV, METEO_STATION,
                                           int(meteo_year))
            if meteo_data_loaded:
                temps = list(meteo_data_loaded.values())
                st.caption(
                    f"✅ {len(meteo_data_loaded)} jours chargés — "
                    f"T° : min {min(temps):.1f}°C / "
                    f"moy {sum(temps)/len(temps):.1f}°C / "
                    f"max {max(temps):.1f}°C"
                )
                # Mini graphique température
                df_t = pd.DataFrame({
                    "date": list(meteo_data_loaded.keys()),
                    "T (°C)": temps
                })
                st.line_chart(df_t.set_index("date"), height=120,
                              use_container_width=True)
            else:
                st.warning("Données météo non disponibles pour cette année.")
        elif use_meteo:
            st.warning(f"Fichier météo introuvable : `{METEO_CSV}`")
        st.session_state["meteo_data"] = meteo_data_loaded
        st.markdown("---")
        # Sélecteur de plage de dates
        st.markdown("#### 📅 Période simulée")
        c1, c2 = st.columns(2)
        with c1:
            date_start = st.date_input(
                "Date de début",
                value=datetime.date(2024, 1, 1),
                min_value=datetime.date(2024, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="date_start",
            )
        with c2:
            date_end = st.date_input(
                "Date de fin",
                value=datetime.date(2024, 12, 31),
                min_value=datetime.date(2024, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="date_end",
            )
        if date_end <= date_start:
            st.error("La date de fin doit être après la date de début.")
        else:
            n_total = (date_end - date_start).days + 1
            vacances_now = st.session_state.get("vacances", VACANCES_DEFAULT)
            # Comptage des jours par type
            n_s = n_w = n_v = 0
            n_hiver = n_printemps = n_ete = n_automne = 0
            cur = date_start
            while cur <= date_end:
                dt = classify_day(cur, vacances_now)
                if dt == "semaine":   n_s += 1
                elif dt == "weekend": n_w += 1
                else:                 n_v += 1
                s = get_season(cur)
                if s == "hiver":       n_hiver += 1
                elif s == "printemps": n_printemps += 1
                elif s == "été":       n_ete += 1
                else:                  n_automne += 1
                cur += datetime.timedelta(days=1)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", f"{n_total} jours")
            col2.metric("Semaine / Weekend / Vac.",
                        f"{n_s} / {n_w} / {n_v}")
            col3.metric("Hiver / Printemps",
                        f"{n_hiver} / {n_printemps}")
            col4.metric("Été / Automne",
                        f"{n_ete} / {n_automne}")
            # Aperçu calendrier coloré (3 mois max)
            if n_total <= 92:
                st.markdown("#### Aperçu du calendrier")
                cal_html = _render_calendar(date_start, date_end, vacances_now)
                st.markdown(cal_html, unsafe_allow_html=True)
        seed = st.number_input("Graine aléatoire (0 = aléatoire)", 0, 9999, 42)
        if seed > 0:
            random.seed(int(seed))
            np.random.seed(int(seed))
        if st.button("🚀 Lancer la simulation", type="primary"):
            if date_end <= date_start:
                st.error("Corrige les dates avant de lancer.")
            else:
                cat_ov   = st.session_state.get("catalogue_override", {})
                forced   = st.session_state.get("forced_events", [])
                members  = st.session_state.get("members", [])
                l_scale  = st.session_state.get("lights_scale", 1.0)
                bf_W     = st.session_state.get("bruit_fond_W", 30)
                vacances = st.session_state.get("vacances", VACANCES_DEFAULT)
                if not cat_ov:
                    st.warning("Configure d'abord les équipements dans l'onglet ⚙️.")
                else:
                    meteo_data    = st.session_state.get("meteo_data", {})
                    weekend_prms  = st.session_state.get("weekend_params", {})
                    af_override   = st.session_state.get("age_factors_override", {})
                    # Recharge les cycles avec calibrage (option 3bis + offset bruit)
                    dur_ov  = {u: c["dur_filter"]   for u, c in cat_ov.items()
                               if c.get("dur_filter")}
                    peak_ov = {u: c["peak_filter"]  for u, c in cat_ov.items()
                               if c.get("peak_filter")}
                    cycles_data_run = load_all_cycles(
                        dur_overrides=dur_ov,
                        peak_overrides=peak_ov,
                        cat_overrides=cat_ov,
                    )
                    with st.spinner(f"Simulation de {n_total} jours en cours…"):
                        results = run_simulation(
                            date_start=date_start,
                            date_end=date_end,
                            members=members,
                            catalogue_override=cat_ov,
                            cycles_data=cycles_data_run,
                            forced_events=forced,
                            lights_scale=l_scale,
                            bruit_fond_W=bf_W,
                            vacances_periods=vacances,
                            meteo_data=meteo_data if meteo_data else None,
                            weekend_params=weekend_prms if weekend_prms else None,
                            age_factors_override=af_override if af_override else None,
                        )
                    st.session_state["sim_results"] = results
                    st.success(f"✅ {n_total} jours simulés — voir 📊 Résultats")

    # ── Onglet Résultats ──────────────────────────────────────────────────────
    with tab_res:
        results = st.session_state.get("sim_results")
        if results is None:
            st.info("Lance d'abord la simulation dans l'onglet 📅.")
        else:
            cat_ov = st.session_state.get("catalogue_override", {})
            # Onglets résultats
            r_tab1, r_tab2 = st.tabs(["📆 Par type de journée", "🍂 Par saison"])
            with r_tab1:
                selected = st.multiselect(
                    "Courbes à afficher",
                    ["semaine", "weekend", "vacances"],
                    default=["semaine", "weekend", "vacances"],
                    key="sel_dtype",
                )
                st.plotly_chart(plot_curves(results, selected),
                                use_container_width=True)
                st.subheader("Statistiques par type de journée")
                st.dataframe(compute_stats(results), use_container_width=True,
                             hide_index=True)
                st.subheader("Décomposition par usage")
                dtype_stack = st.radio("Type de journée",
                                       ["semaine", "weekend", "vacances"],
                                       horizontal=True, key="stack_dtype")
                st.plotly_chart(plot_stackplot(results, dtype_stack, cat_ov),
                                use_container_width=True)
            with r_tab2:
                saisons_dispo = [s for s in ["hiver", "printemps", "été", "automne"]
                                 if results.get(f"n_{s}", 0) > 0]
                sel_saisons = st.multiselect(
                    "Saisons à afficher",
                    saisons_dispo,
                    default=saisons_dispo,
                    key="sel_saison",
                )
                if sel_saisons:
                    st.plotly_chart(plot_curves(results, sel_saisons),
                                    use_container_width=True)
                    st.caption(
                        "ℹ️ Les moyennes par saison excluent les jours de vacances "
                        "pour éviter de biaiser la courbe vers le bruit de fond."
                    )
                    # Stats saisons avec détail semaine/weekend
                    rows = []
                    emojis = {"hiver":"❄️","printemps":"🌸","été":"☀️","automne":"🍂"}
                    for s in saisons_dispo:
                        mean   = results.get(f"mean_{s}",          np.zeros(MINUTES_DAY))
                        mean_s = results.get(f"mean_{s}_semaine",   np.zeros(MINUTES_DAY))
                        mean_w = results.get(f"mean_{s}_weekend",   np.zeros(MINUTES_DAY))
                        n_s    = results.get(f"n_{s}_semaine", 0)
                        n_w    = results.get(f"n_{s}_weekend", 0)
                        rows.append({
                            "Saison":            f"{emojis.get(s,'')} {s.capitalize()}",
                            "Jours (sem/we)":    f"{n_s}/{n_w}",
                            "Moy. globale (W)":  round(mean.mean(),   1),
                            "Moy. semaine (W)":  round(mean_s.mean(),  1) if n_s else "—",
                            "Moy. weekend (W)":  round(mean_w.mean(),  1) if n_w else "—",
                            "Pic (W)":           round(mean.max(),    1),
                            "Énergie/jour (kWh)":round(mean.sum()/60_000, 3),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)
                    st.subheader("Décomposition par usage")
                    saison_stack = st.radio("Saison",
                                            saisons_dispo, horizontal=True,
                                            key="stack_saison")
                    st.plotly_chart(
                        plot_stackplot(results, saison_stack, cat_ov),
                        use_container_width=True
                    )

            # ── Diagnostic usage par saison ────────────────────────────────
            st.subheader("🔍 Diagnostic — Puissance moyenne par usage et par saison")
            st.caption("Moyennes en W · Vacances exclues des moyennes saisonnières")
            saisons_all = ["hiver", "printemps", "été", "automne"]
            emojis_d = {"hiver":"❄️","printemps":"🌸","été":"☀️","automne":"🍂"}
            # Collecte tous les usages présents dans au moins une contrib
            all_usages = set()
            for s in saisons_all:
                all_usages.update(results.get(f"contrib_{s}", {}).keys())
            for dt in ["semaine", "weekend", "vacances"]:
                all_usages.update(results.get(f"contrib_{dt}", {}).keys())
            all_usages = sorted(all_usages)
            LABELS_USAGE = {u: CATALOGUE.get(u, {}).get("label", u)
                            for u in all_usages}
            diag_rows = []
            for u in all_usages:
                # Moyenne annuelle = moyenne sur tous les jours (sem+we, hors vac)
                vals_ann = []
                for s in saisons_all:
                    c = results.get(f"contrib_{s}", {}).get(u)
                    if c is not None:
                        vals_ann.append(float(c.mean()))
                moy_ann = round(np.mean(vals_ann), 2) if vals_ann else 0.0
                # Moyenne par saison
                moy_s = {}
                for s in saisons_all:
                    c = results.get(f"contrib_{s}", {}).get(u)
                    moy_s[s] = round(float(c.mean()), 2) if c is not None else 0.0
                # Moyenne annuelle attendue = somme saisons / 4
                moy_attendue = round(sum(moy_s.values()) / 4, 2)
                diag_rows.append({
                    "Usage": LABELS_USAGE.get(u, u),
                    "Moy. annuelle (W)":  moy_ann,
                    f"{emojis_d['hiver']} Hiver (W)":      moy_s["hiver"],
                    f"{emojis_d['printemps']} Printemps (W)": moy_s["printemps"],
                    f"{emojis_d['été']} Été (W)":         moy_s["été"],
                    f"{emojis_d['automne']} Automne (W)":  moy_s["automne"],
                    "Attendue∑/4 (W)":   moy_attendue,
                    "Écart (W)":         round(moy_ann - moy_attendue, 2),
                })
            df_diag = pd.DataFrame(diag_rows).sort_values(
                "Moy. annuelle (W)", ascending=False
            ).reset_index(drop=True)
            # Surligne les écarts importants

            def color_ecart(val):

                if isinstance(val, float) and abs(val) > 5:
                    return "background-color: #fff3cd"
                return ""
            st.dataframe(
                df_diag.style.applymap(color_ecart, subset=["Écart (W)"]),
                use_container_width=True, hide_index=True
            )
            csv_diag = df_diag.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Télécharger diagnostic CSV", csv_diag,
                "diagnostic_usages.csv", "text/csv"
            )
            # Export CSV courbes
            st.subheader("Export courbes")
            for dtype in ["semaine", "weekend", "vacances"]:
                mean = results.get(f"mean_{dtype}", np.zeros(MINUTES_DAY))
                df_export = pd.DataFrame({
                    "time_min": np.arange(MINUTES_DAY),
                    "power_W":  mean.round(2),
                })
                csv_bytes = df_export.to_csv(index=False).encode()
                st.download_button(
                    label=f"⬇️ Télécharger courbe {dtype} (CSV)",
                    data=csv_bytes,
                    file_name=f"courbe_{dtype}.csv",
                    mime="text/csv",
                )
if __name__ == "__main__":
    main()