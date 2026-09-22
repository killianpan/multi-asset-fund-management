# =============================================================================
# INTRODUCTION
# =============================================================================


"""
Module de mise à jour de la base de données du fond d'investissement.
Appelé chaque lundi lorsqu'une stratégie génère des ordres d'achat ou de vente.
Actualise les tables Deals et Portfolios en présence de nouvelles décisions et affiche une synthèse des ordres hebdomadaires passés pour chaque portefeuille.
"""

from datetime import date
import sqlite3
import pandas as pd
import json
from modules.strategies import strategy_low_risk, strategy_low_turnover, strategy_high_yield_equity_only
from modules.strategies import get_current_holdings


# =============================================================================
# SAUVEGARDE DES DEALS
# =============================================================================


def save_deals(deals: list[dict], db_path: str = "db/Fund.db") -> None:
    
    """
    Insère les deals hebdomadaires retournés par une stratégie dans la table Deals de la base de données.
    """
    
    if not deals:
        return

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO Deals (portfolio_id, date, ticker, action)
        VALUES (:portfolio_id, :date, :ticker, :action)
    """, deals)

    conn.commit()
    conn.close()


# =============================================================================
# ACTUALISATION DU PORTEFEUILLE
# =============================================================================


def update_portfolio_assets(portfolio_id: int, db_path: str = "db/Fund.db") -> None:
    
    """
    Actualise la composition du portefeuille donné à l'aide des ordres d'achat et de vente passés en début de semaine.
    """

    # Récupération des actifs en portefeuille à partir des deals passés
    current_holdings = get_current_holdings(portfolio_id, db_path)

    # Conversion en JSON (liste)
    assets_json = json.dumps(list(current_holdings))

    # Actualisation de la table Portfolios
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE Portfolios
        SET assets = ?
        WHERE client_id = ?
    """, (assets_json, portfolio_id))

    conn.commit()
    conn.close()


# =============================================================================
# AFFICHAGE FORMATÉ — FORMAT SUJET
# =============================================================================
  
    
def print_decisions(deals: list[dict], portfolio_name: str, date: pd.Timestamp, portfolio_id: int) -> None:
    
    """
    Affiche les décisions d'investissement de la semaine pour le portefeuille donné dans le format demandé par le sujet :
    "Lundi 22/01 : portefeuille 1 "Low Risk" acheter AAPL, vendre TLT"
    """

    date_str = date.strftime("%d/%m")
    prefix   = f'  Lundi {date_str} : portefeuille {portfolio_id} "{portfolio_name}" -> '

    if not deals:
        print(prefix + "aucun deal")
        return

    buys  = [d["ticker"] for d in deals if d["action"] == "BUY"]
    sells = [d["ticker"] for d in deals if d["action"] == "SELL"]

    parts = []
    if buys:  parts.append(f"acheter {', '.join(buys)}")
    if sells: parts.append(f"vendre {', '.join(sells)}")

    print(prefix + " | ".join(parts))


# =============================================================================
# FONCTION DE SYNTHESE : EXÉCUTION DES STRATÉGIES + SAUVEGARDE + AFFICHAGE
# =============================================================================


def run_monday_update(close_prices: pd.DataFrame, date: pd.Timestamp, db_path: str = "db/Fund.db") -> None:
    
    """
    Orchestre l'exécution des 3 stratégies et la sauvegarde des décisions d'investissement dans la base de données pour un lundi donné.
    """

    print(f"\n {date.strftime('%A %d/%m/%Y')}")

    # Stratégie 1 : Low Risk
    deals_lr = strategy_low_risk(close_prices, date, db_path)
    save_deals(deals_lr, db_path)
    update_portfolio_assets(1, db_path)
    print_decisions(deals_lr, "Low Risk", date, portfolio_id=1)

    # Stratégie 2 : Low Turnover
    deals_lt = strategy_low_turnover(close_prices, date, db_path)
    save_deals(deals_lt, db_path)
    update_portfolio_assets(2, db_path)
    print_decisions(deals_lt, "Low Turnover", date, portfolio_id=2)

    # Stratégie 3 : High Yield Equity Only
    deals_hy = strategy_high_yield_equity_only(close_prices, date, db_path)
    save_deals(deals_hy, db_path)
    update_portfolio_assets(3, db_path)
    print_decisions(deals_hy, "High Yield Equity Only", date, portfolio_id=3)