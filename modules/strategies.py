# =============================================================================
# INTRODUCTION
# =============================================================================


"""
Module contenant les stratégies de gestion de portefeuille administrées par le fond multi-asset.

Trois profils de gestion sont implémentés, chacun avec une approche différente :

- Low Risk       : Minimisation de la volatilité annualisée avec plafond de 10%
- Low Turnover   : Limite de 2 deals par mois, rotation basée sur le ratio de Sharpe 
- High Yield EQ  : Maximisation du rendement  sur le marché action uniquement, avec ajustement dynamique selon le VIX
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from scipy.optimize import minimize


# =============================================================================
# CONSTANTES
# =============================================================================


EQUITY_TICKERS = [
"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
    "JPM", "JNJ", "PG", "XOM", "KO", "WMT", "TSLA"
]

ALL_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "JNJ", "PG", "XOM",
    "KO", "WMT", "TSLA", "TLT", "IEF", "AGG", "LQD", "JNK",
    "USO", "UNG", "DBC", "GLD", "SPY", "^GSPC", "^VIX", "^TNX"
]

VOL_TARGET      = 0.10      # Cible de volatilité annualisée (Low Risk)
MOM_WINDOW      = 60        # Fenêtre momentum (jours)
TRADING_DAYS    = 252       # Jours de trading annuels
MAX_WEIGHT      = 0.30      # Poids maximum par actif
MIN_WEIGHT      = 0.05      # Seuil minimum pour considérer qu'on détient un actif
MAX_DEALS_PER_MONTH = 2     # Limite de deals mensuels (Low Turnover)
INITIAL_PORTFOLIO_SIZE = 6  # Nombre d'actifs à constituer au départ


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_past_returns(close_prices: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    
    """
    Renvoie les rendements journaliers passés strictement de l'univers d'investissement à l'activation d'une stratégie donnée (date)
    Garantit l'absence de look-forward bias.
    """

    past_prices = close_prices.loc[close_prices.index < date]
    returns = past_prices.pct_change().dropna()
    
    return returns


def get_current_holdings(portfolio_id: int, db_path: str = "db/Fund.db") -> set:
    
    """
    Renvoie les positions actuelles du portefeuille donné depuis la table Deals de la base de données.
    Seules les plus récentes positions "BUY" sont considérées comme détenues.
    """

    conn = sqlite3.connect(db_path)
    query = """
        SELECT ticker, action
        FROM Deals
        WHERE portfolio_id = ?
        ORDER BY date ASC, deal_id ASC
    """
    df = pd.read_sql(query, conn, params=(portfolio_id,))
    conn.close()

    holdings = {}
    for _, row in df.iterrows():
        holdings[row["ticker"]] = row["action"]

    # Renvoie uniquement les ordres d'achat qui n'ont pas encore de contrepartie en vente, càd les actifs en portefeuille
    return {t for t, a in holdings.items() if a == "BUY"}


def generate_deals(new_holdings: list, current_holdings: set, portfolio_id: int, date: pd.Timestamp) -> list[dict]:
    
    """
    A l'activation hebdomadaire de le stratégie donnée, renvoie la liste des deals à exécuter pour passer du portefeuille actuel au nouveau portefeuille recommandé.
    """

    date_str = date.strftime("%Y-%m-%d")
    new_set  = set(new_holdings)

    # Reconstitue les ordres d'achat et de vente à ajouter ultérieurement dans la base de données
    sells = [{"portfolio_id": portfolio_id, "date": date_str, "ticker": t, "action": "SELL"}
             for t in current_holdings - new_set]
    buys  = [{"portfolio_id": portfolio_id, "date": date_str, "ticker": t, "action": "BUY"}
             for t in new_set - current_holdings]

    return sells + buys


def get_monthly_deal_count(portfolio_id: int, date: pd.Timestamp, db_path: str) -> int:
    
    """
    Fonction spécifique à la stratégie Low Turnover 
    Renvoie le nombre de deals déjà exécutés depuis le début du mois jusqu'à la date actuelle.
    """
    
    conn = sqlite3.connect(db_path)

    start_month = date.replace(day=1).strftime("%Y-%m-%d")
    end_month   = date.strftime("%Y-%m-%d")

    query = """
        SELECT COUNT(*) as n
        FROM Deals
        WHERE portfolio_id = ?
        AND date >= ?
        AND date <= ?
    """

    df = pd.read_sql(query, conn, params=(portfolio_id, start_month, end_month))
    conn.close()

    return int(df["n"].iloc[0])


# =============================================================================
# STRATÉGIE 1 — LOW RISK
# Objectif : volatilité annualisée ~10% en minimisant la variance du portefeuille
# Portfolio ID : 1
# =============================================================================


def strategy_low_risk(close_prices: pd.DataFrame, date: pd.Timestamp, db_path: str = "db/Fund.db") -> list[dict]:
    
    """
    Stratégie Low Risk — Minimum Variance avec optimisation de Markowitz.

    Étapes :
    1. Calcul de la matrice de covariance sur VOL_WINDOW jours passés
    2. Optimisation de Markowitz pour minimiser la variance du portefeuille
       sous contrainte de somme des poids égale à 1
    3. Mise à l'échelle des poids pour cibler VOL_TARGET (~10% vol annualisée)
    4. Le cash résiduel est alloué à TLT (ou l'actif le moins volatil si TLT indisponible)
    5. Sélection des actifs à détenir (poids > MIN_WEIGHT)
    6. Comparaison avec les positions actuelles → génération des deals

    Renvoie la liste des deals à exécuter, la constitution du portefeuille, le portefeuille concerné par la stratégie et la date d'exécution
    """

    returns = get_past_returns(close_prices, date)

    if len(returns) < MOM_WINDOW:
        return []

    # Exclure les indices non tradeables (^GSPC, ^VIX, ^TNX)
    tradeable_tickers = [t for t in ALL_TICKERS if t not in ["^GSPC", "^VIX", "^TNX"]]

    # Filtrer les actifs disponibles sur la fenêtre
    recent = returns[[t for t in tradeable_tickers if t in returns.columns]].tail(MOM_WINDOW).dropna(axis=1)

    if recent.shape[1] < 2:
        return []

    # Calcul de la matrice de covariance annualisée
    cov_matrix = recent.cov() * TRADING_DAYS

    # Fonction objectif : minimiser la variance du portefeuille
    def portfolio_variance(weights):
        return weights.T @ cov_matrix.values @ weights

    # Contraintes : somme des poids = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bornes : chaque poids entre 0 et MAX_WEIGHT
    bounds = [(0, MAX_WEIGHT) for _ in range(len(cov_matrix))]

    # Initialisation des poids égaux
    initial_weights = np.array([1 / len(cov_matrix)] * len(cov_matrix))

    # Optimisation quadratique
    result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)

    if not result.success:
        return []

    optimized_weights = pd.Series(result.x, index=cov_matrix.columns)

    # Volatilité annualisée du portefeuille optimisé
    port_vol = np.sqrt(portfolio_variance(optimized_weights.values))

    # Mise à l'échelle vers VOL_TARGET (sans effet de levier : scale ≤ 1)
    scale = min(VOL_TARGET / port_vol, 1.0) if port_vol > 0 else 1.0
    scaled_weights = optimized_weights * scale

    # Fonction utilitaire : alloue le résidu à TLT ou à l'actif le moins volatil
    def allocate_residual(scaled, residual):
        if residual <= 0:
            return scaled
        refuge = "TLT" if "TLT" in scaled.index else cov_matrix.var(axis=0).idxmin()
        scaled[refuge] = scaled.get(refuge, 0) + residual
        return scaled

    # Résidu initial (capital non investi après scaling)
    scaled_weights = allocate_residual(scaled_weights, 1.0 - scaled_weights.sum())

    # Plafonnement à MAX_WEIGHT puis résidu du plafonnement → refuge
    scaled_weights = scaled_weights.clip(upper=MAX_WEIGHT)
    scaled_weights = allocate_residual(scaled_weights, 1.0 - scaled_weights.sum())

    # Sélection des actifs à détenir et génération des deals
    new_holdings = scaled_weights[scaled_weights >= MIN_WEIGHT].index.tolist()
    current_holdings = get_current_holdings(1, db_path)

    return generate_deals(new_holdings, current_holdings, 1, date)


# =============================================================================
# STRATÉGIE 2 — LOW TURNOVER
# Objectif : max 2 deals par mois - maximisation du ratio de Sharpe sur les 30 derniers jours
# Portfolio ID : 2
# =============================================================================

def strategy_low_turnover(close_prices: pd.DataFrame, date: pd.Timestamp, db_path: str = "db/Fund.db") -> list[dict]:
   
    """
    Stratégie Low Turnover : Max 2 deals par mois

    Etapes :
    1. Construction du portefeuille sur la base du top 6 des actifs selon le momentum 60 jours
    2. Si portefeuille déjà constitué, évaluation du ratio de Sharpe sur les 30 derniers jours pour les actifs détenus et non détenus
    3. Si le maximum des deals atteint, aucune modification du portefeuille
    3. Sinon, proposition d'un échange hebdomadaire : vendre l'actif détenu avec le pire Sharpe et acheter l'actif non détenu avec le meilleur Sharpe

    Renvoie la liste des deals à exécuter, la constitution du portefeuille, le portefeuille concerné par la stratégie et la date d'exécution
    """

    # Etape 1 : Construction du portefeuille
    returns = get_past_returns(close_prices, date)

    # Récupération des rendements passés des actifs tradeables
    tradeable_tickers = [t for t in ALL_TICKERS if t not in ["^GSPC", "^VIX", "^TNX"]]
    recent_returns = returns[[t for t in tradeable_tickers if t in returns.columns]]

    if recent_returns.shape[1] < 2 or len(recent_returns) < 5:
        return []

    # Portefeuille actuel et deals mensuels
    portfolio_id = 2
    current_holdings = get_current_holdings(portfolio_id, db_path)
    monthly_deals = get_monthly_deal_count(portfolio_id, date, db_path)
    remaining_deals = max(0, 2 - monthly_deals)

    # Si le nombre maximum de deals pour le mois est atteint, ne pas modifier le portefeuille
    if remaining_deals == 0:
        return []

    #  Portefeuille initial (momentum 60 jours)
    if len(current_holdings) == 0:
        momentum_60d = recent_returns.tail(60).sum()
        n_buy = min(INITIAL_PORTFOLIO_SIZE, len(momentum_60d))
        new_holdings = momentum_60d.sort_values(ascending=False).index[:n_buy].tolist()

        return generate_deals(new_holdings, current_holdings, portfolio_id, date)

    # Etape 2 : Rotation basée sur le ratio de Sharpe (30 derniers jours)
    new_holdings = set(current_holdings)

    recent_30d = recent_returns.tail(30)

    # Calcul du ratio de Sharpe pour chaque actif
    mean_returns = recent_30d.mean()
    vol_returns = recent_30d.std()
    sharpe = mean_returns / vol_returns
    sharpe = sharpe.replace([np.inf, -np.inf], 0).fillna(0)

    # Sélection de l'actif détenu avec le pire Sharpe et de l'actif non détenu avec le meilleur Sharpe
    worst_current = sharpe.loc[list(current_holdings)].idxmin()
    best_new = sharpe.drop(current_holdings).idxmax()

    # Echange si worst_current > best_new
    if sharpe[best_new] > sharpe[worst_current]:
        new_holdings.remove(worst_current)
        new_holdings.add(best_new)

    return generate_deals(list(new_holdings), current_holdings, portfolio_id, date)

# =============================================================================
# STRATÉGIE 3 — HIGH YIELD EQUITY ONLY
# Objectif : maximiser le rendement, actions uniquement, aucune contrainte
# Portfolio ID : 3
# =============================================================================

def strategy_high_yield_equity_only(close_prices: pd.DataFrame, date: pd.Timestamp, db_path: str = "db/Fund.db") -> list[dict]:
   
    """
    Stratégie High Yield Equity Only — Momentum + allocation dynamique.

    Etapes :
    1. Calcul du momentum sur les 60 derniers jours des actions de l'univers d'investissement
    2. Ajustement de la concentration selon le VIX : plus l'incertitude du marché est élevée, plus la diversification est forte 
    3. Sélection des actifs les plus performants, avec un poids amplifié pour les meilleurs actifs, en respectant la contrainte de diversification

    Renvoie la liste des deals à exécuter, la constitution du portefeuille, le portefeuille concerné par la stratégie et la date d'exécution
    """

    # Etape 1 : Calcul du momentum sur les 60 derniers jours
    returns = get_past_returns(close_prices, date)

    if len(returns) < 60:
        return []

    # Filtrer les actions
    equity_returns = returns[[t for t in EQUITY_TICKERS if t in returns.columns]].tail(60)

    if equity_returns.shape[1] < 2:
        return []

    # Score momentum
    momentum = equity_returns.mean()
    momentum = momentum[momentum > 0]
    if momentum.empty:
        return []

    # Etape 2 : Ajustement de la concentration selon le VIX
    if "^VIX" in returns.columns:
        vix = returns["^VIX"].iloc[-1]
    else:
        vix = 20  # valeur neutre

    #Ajustement de la concentration
    if vix > 25:
        max_weight = 0.20
    elif vix < 15:
        max_weight = 0.40
    else:
        max_weight = 0.30

    # Etape 3 : sélection des actifs les plus performants

    # Softmax pour amplifier les différences   
    alpha = 10
    exp_scores = np.exp(alpha * momentum)
    weights = exp_scores / exp_scores.sum()

    # Contrainte max_weight
    weights = weights.clip(upper=max_weight)

    # Renormalisation
    weights = weights / weights.sum()

    # Sélection des actifs
    new_holdings = weights[weights >= MIN_WEIGHT].index.tolist()

    # Génération des deals
    current_holdings = get_current_holdings(3, db_path)

    return generate_deals(new_holdings, current_holdings, 3, date)