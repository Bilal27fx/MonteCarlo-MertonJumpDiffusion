import streamlit as st
import numpy as np
import plotly.graph_objects as go
from Simulation import display_option  # Si la classe est dans un fichier externe, importez-la

# Main function to call display_option
def main():
    # Paramètres par défaut
    S0 = 100  # Prix initial de l'actif
    r = 0.05  # Taux sans risque
    sigma = 0.2  # Volatilité
    T = 1.0  # Maturité
    k = 100  # Prix d'exercice
    n_simulations = 10000  # Nombre de simulations
    lambda_jump = 0.2  # Intensité des sauts
    mu_jump = -0.15  # Moyenne des sauts
    sigma_jump = 0.1  # Volatilité des sauts
    n_pas = 100  # Nombre de pas
    k_values = np.linspace(50, 200, 30)  # Valeurs pour K (Strike Price)
    T_values = np.linspace(0.1, 5.0, 30)  # Valeurs pour T (Maturité)

    # Créer l'instance de la classe display_option
    display = display_option(S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_pas, k_values, T_values)
    
    # Appeler la méthode pour afficher la simulation des options
    display.display_option_simulation()

# Appel de la fonction principale
if __name__ == "__main__":
    main()
