# =============================================================================
# INTRODUCTION
# =============================================================================


"""
Module de collecte et de nettoyage des données financières pour la construction de portefeuilles d'investissement.

Ce module comprend deux fonctions principales :
1. download_prices: Télécharge les prix de clôture ajustés pour une liste de tickers et sur une période données à l'aide de l'API Yahoo Finance. Les données brutes sont sauvegardées dans un fichier CSV.
2. clean_prices: Nettoie le jeu de données en remplissant les valeurs manquantes par la méthode ffill et en appliquant une winsorisation pour limiter l'impact des valeurs extrêmes. Les données nettoyées sont sauvegardées dans un fichier CSV.
"""

import yfinance as yf
import pandas as pd
import os


# =============================================================================
# TELECHARGEMENT DES DONNEES
# =============================================================================

def download_prices(tickers, start, end, save_raw=True):

    data = yf.download(tickers, start=start, end=end, group_by="column")

    # Gestion des cas multi-index (plusieurs tickers) ou simple index (un ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            data = data["Adj Close"]
        else:
            data = data["Close"]
    else:
        data = data.to_frame(name=tickers[0])

    # Sauvegarde
    if save_raw:
        os.makedirs("data/raw", exist_ok=True)
        data.to_csv("data/raw/prices_raw.csv")

    return data


# =============================================================================
# NETTOYAGE DES DONNEES
# =============================================================================


def clean_prices(price_df, save_processed=True):
    
    # Remplir les trous uniquement avec ffill 
    price_df = price_df.ffill()

    # Colonnes macro à laisser intactes
    macro_cols = ["^VIX", "^TNX"]

    # Séparer colonnes macro et colonnes financières
    financial_cols = [col for col in price_df.columns if col not in macro_cols]
    macro_data = price_df[macro_cols].copy()
    financial_data = price_df[financial_cols].copy()

    # Calcul des rendements
    returns = financial_data.pct_change()

    # Winsorisation colonne par colonne
    for col in returns.columns:
        col_mean = returns[col].mean()
        col_std = returns[col].std()
        z = (returns[col] - col_mean) / col_std

        # Remplace les valeurs extrêmes par la moyenne de la colonne
        returns.loc[z > 4, col] = col_mean
        returns.loc[z < -4, col] = col_mean

    # Reconstruction des prix nettoyés
    clean_prices = (1 + returns).cumprod() * financial_data.iloc[0]

    # Concaténer colonnes macro intactes et colonnes financières nettoyées
    clean_prices_df = pd.concat([clean_prices, macro_data], axis=1)
    clean_prices_df = clean_prices_df.drop(clean_prices_df.index[0])

    # Sauvegarde
    if save_processed:
        os.makedirs("data/processed", exist_ok=True)
        clean_prices_df.to_csv("data/processed/prices_clean.csv")

    return clean_prices_df


