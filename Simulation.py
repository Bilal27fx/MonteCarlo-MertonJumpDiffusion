import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

class MonteCarloSimulator_BlackScholes:
    def __init__(self, S0, r, sigma, T, k, n_simulations):
        self.S0 = S0  # Prix initial de l'actif
        self.r = r  # Taux sans risque
        self.sigma = sigma  # Volatilité
        self.T = T  # Maturité
        self.k = k  # Prix d'exercice (strike)
        self.n_simulations = n_simulations  # Nombre de simulations

    def Call_Price_BlackScholes(self):
        Wt = np.random.normal(0, np.sqrt(self.T), self.n_simulations)  
        St = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * Wt)  
        payoffs = np.maximum(St - self.k, 0)  
        CallPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  
        return CallPrice

    def Put_Price_BlackScholes(self):
        Wt = np.random.normal(0, np.sqrt(self.T), self.n_simulations)
        St = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * Wt)
        payoffs = np.maximum(self.k - St, 0)  
        PutPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  
        return PutPrice


class MonteCarloSimulator_MertonJumpDiffusion:
    def __init__(self, S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_pas):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.k = k
        self.n_simulations = n_simulations
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.n_pas = n_pas

    def Simulate_Jump_Diffusion(self):
        dt = self.T / self.n_pas  # Intervalle de temps
        St = np.zeros((self.n_simulations, self.n_pas + 1))  # Matrice pour stocker les trajectoires
        St[:, 0] = self.S0  # Initialiser le prix initial
        for i in range(self.n_simulations):
            # Simulation du mouvement brownien et des sauts
            Wt = np.random.normal(0, np.sqrt(dt), self.n_pas)  # Mouvement brownien
            N = np.random.poisson(self.lambda_jump * dt, self.n_pas)  # Nombre de sauts
            jumps = np.random.normal(self.mu_jump, self.sigma_jump, self.n_pas)  # Amplitude des sauts
            # Simuler le prix de l'actif avec sauts
            for t in range(1, self.n_pas + 1):
                St[i, t] = St[i, t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * Wt[t - 1] + jumps[t - 1] * N[t - 1])
        return St

    def Call_Price_MertonJumpDiffusion(self):
        St = self.Simulate_Jump_Diffusion()  # Simuler les trajectoires avec sauts
        payoffs = np.maximum(St[:, -1] - self.k, 0)  # Calcul des payoffs de l'option Call
        CallPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  # Actualisation du prix de l'option Call
        return CallPrice

    def Put_Price_MertonJumpDiffusion(self):
        St = self.Simulate_Jump_Diffusion()  # Simuler les trajectoires avec sauts
        payoffs = np.maximum(self.k - St[:, -1], 0)  # Calcul des payoffs de l'option Put
        PutPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  # Actualisation du prix de l'option Put
        return PutPrice


class display_option:
    def __init__(self, S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_pas, k_values, T_values):
        self.S0 = S0  # Prix initial de l'actif
        self.r = r  # Taux sans risque
        self.sigma = sigma  # Volatilité
        self.T = T  # Maturité
        self.k = k  # Prix d'exercice (strike)
        self.n_simulations = n_simulations  # Nombre de simulations
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.n_pas = n_pas
        self.k_values = k_values
        self.T_values = T_values

    def calculate_option_prices(self, S0, r, sigma, T, k_values, n_simulations, option_type="call"):
        prices = np.zeros((len(k_values), len(T)))
        for i, T_val in enumerate(T):
            for j, k_val in enumerate(k_values):
                # Choisir la méthode de simulation en fonction de l'option (Black-Scholes ou Merton Jump Diffusion)
                if option_type == "call" or option_type == "put":
                    bs_simulator = MonteCarloSimulator_BlackScholes(S0, r, sigma, T_val, k_val, n_simulations)
                    if option_type == "call":
                        option_price = bs_simulator.Call_Price_BlackScholes()
                    else:
                        option_price = bs_simulator.Put_Price_BlackScholes()
                prices[j, i] = option_price
        return prices

    def display_option_simulation(self):
        # Streamlit interface configuration
        st.title("Monte Carlo Simulation for Options")
        
        # Sidebar for Black-Scholes parameters
        st.sidebar.header("Black-Scholes Option Parameters")
        S0 = st.sidebar.slider("Initial Price", 50, 200, 100)
        r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.05)
        sigma = st.sidebar.slider("Volatility", 0.0, 1.0, 0.2)
        T = st.sidebar.slider("Maturity", 0.1, 10.0, 1.0)
        k = st.sidebar.slider("Strike Price", 50, 200, 100)
        n_simulations = st.sidebar.slider("Number of Simulations", 1000, 3000, 1000)

        # Sidebar for Merton Jump Diffusion parameters
        st.sidebar.header("Merton Jump Diffusion Option Parameters")
        lambda_jump = st.sidebar.slider("Jump Intensity", 0.0, 1.0, 0.2)
        mu_jump = st.sidebar.slider("Jump Mean", -1.0, 0.0, -0.15)
        sigma_jump = st.sidebar.slider("Jump Volatility", 0.0, 1.0, 0.1)
        n_steps = st.sidebar.slider("Number of Time Steps", 100, 500, 100)

        # Create simulators
        bs_simulator = MonteCarloSimulator_BlackScholes(S0, r, sigma, T, k, n_simulations)
        mj_simulator = MonteCarloSimulator_MertonJumpDiffusion(S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_steps)

        # Calculate option prices for both methods
        call_price_bs = bs_simulator.Call_Price_BlackScholes()
        put_price_bs = bs_simulator.Put_Price_BlackScholes()
        call_price_mj = mj_simulator.Call_Price_MertonJumpDiffusion()
        put_price_mj = mj_simulator.Put_Price_MertonJumpDiffusion()

        # Affichage des résultats dans 2 colonnes
        st.subheader("Option Prices for Black-Scholes Model")

        # Affichage des prix sous forme de carrés colorés
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 8px; text-align: center;">Call Price: <strong>{call_price_bs:.2f}</strong></div>',
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div style="background-color: #f44336; color: white; padding: 10px; border-radius: 8px; text-align: center;">Put Price: <strong>{put_price_bs:.2f}</strong></div>',
                unsafe_allow_html=True)

        # Affichage des graphiques Black-Scholes avec taille réduite
        col1, col2 = st.columns(2)
        with col1:
            option_prices_bs_call = self.calculate_option_prices(self.S0, self.r, self.sigma, self.T_values, self.k_values, self.n_simulations, option_type="call")
            fig_bs_call = go.Figure(data=[go.Surface(
                z=option_prices_bs_call,
                x=self.k_values,  # x is Strike Price (K)
                y=self.T_values,  # y is Time to Maturity (T)
                colorscale='Jet',  # Heatmap color scale
                colorbar=dict(title='Option Price'),
            )])

            fig_bs_call.update_layout(
                scene=dict(
                    xaxis_title='Strike Price (K)',
                    yaxis_title='Time to Maturity (T)',
                    zaxis_title='Option Price (Call)'
                ),
                height=500,  # Taille réduite
                width=500   # Taille réduite
            )

            st.plotly_chart(fig_bs_call, use_container_width=True)

        with col2:
            option_prices_bs_put = self.calculate_option_prices(self.S0, self.r, self.sigma, self.T_values, self.k_values, self.n_simulations, option_type="put")
            fig_bs_put = go.Figure(data=[go.Surface(
                z=option_prices_bs_put,
                x=self.k_values,  # x is Strike Price (K)
                y=self.T_values,  # y is Time to Maturity (T)
                colorscale='Jet',  # Heatmap color scale
                colorbar=dict(title='Option Price'),
            )])

            fig_bs_put.update_layout(
                scene=dict(
                    xaxis_title='Strike Price (K)',
                    yaxis_title='Time to Maturity (T)',
                    zaxis_title='Option Price (Put)'
                ),
                height=500,  # Taille réduite
                width=500   # Taille réduite
            )

            st.plotly_chart(fig_bs_put, use_container_width=True)

        st.subheader("Option Prices for Merton Jump Diffusion Model")

        # Affichage des prix sous forme de carrés colorés
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f'<div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 8px; text-align: center;">Call Price: <strong>{call_price_mj:.2f}</strong></div>',
                unsafe_allow_html=True)
        with col4:
            st.markdown(
                f'<div style="background-color: #FF9800; color: white; padding: 10px; border-radius: 8px; text-align: center;">Put Price: <strong>{put_price_mj:.2f}</strong></div>',
                unsafe_allow_html=True)

        # Affichage des graphiques Merton Jump Diffusion avec taille réduite
        col3, col4 = st.columns(2)
        with col3:
            option_prices_mj_call = self.calculate_option_prices(self.S0, self.r, self.sigma, self.T_values, self.k_values, self.n_simulations, option_type="call")
            fig_mj_call = go.Figure(data=[go.Surface(
                z=option_prices_mj_call,
                x=self.k_values,  # x is Strike Price (K)
                y=self.T_values,  # y is Time to Maturity (T)
                colorscale='Jet',  # Heatmap color scale
                colorbar=dict(title='Option Price'),
            )])

            fig_mj_call.update_layout(
                scene=dict(
                    xaxis_title='Strike Price (K)',
                    yaxis_title='Time to Maturity (T)',
                    zaxis_title='Option Price (Call)'
                ),
                height=500,  # Taille réduite
                width=500   # Taille réduite
            )

            st.plotly_chart(fig_mj_call, use_container_width=True)

        with col4:
            option_prices_mj_put = self.calculate_option_prices(self.S0, self.r, self.sigma, self.T_values, self.k_values, self.n_simulations, option_type="put")
            fig_mj_put = go.Figure(data=[go.Surface(
                z=option_prices_mj_put,
                x=self.k_values,  # x is Strike Price (K)
                y=self.T_values,  # y is Time to Maturity (T)
                colorscale='Jet',  # Heatmap color scale
                colorbar=dict(title='Option Price'),
            )])

            fig_mj_put.update_layout(
                scene=dict(
                    xaxis_title='Strike Price (K)',
                    yaxis_title='Time to Maturity (T)',
                    zaxis_title='Option Price (Put)'
                ),
                height=500,  # Taille réduite
                width=500   # Taille réduite
            )

            st.plotly_chart(fig_mj_put, use_container_width=True)
