# =============================================================================
# INTRODUCTION
# =============================================================================


"""
Module de création et de peuplement initial de la base de données SQLite du fond multi-assets.

La base de données ainsi créée contient les tables suivantes :
- Clients : Identité des clients et leur profil de risque
- Products : Description des actifs constituant l'univers d'investissement à travers leur ticker, nom et classe d'actif
- Returns : Rendements historiques pour chaque produit
- Portfolios : Composition des portefeuilles clients
- Managers : Informations sur les gestionnaires de portefeuille
- Deals : Historique des transactions effectuées sur les portefeuilles
"""

import sqlite3
import json
import pandas as pd
import os


# =============================================================================
# CREATION DE LA BASE DE DONNEES ET DES TABLES
# =============================================================================

# Création de la base
def create_database(db_path="db/Fund.db"):

    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.close()


# Création des tables
def create_tables(db_path="db/Fund.db"):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Clients
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Clients (
            client_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            risk_profile TEXT
        )
    """)

    # Products
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Products (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            asset_class TEXT
        )
    """)

    # Returns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Returns (
            date TEXT,
            ticker TEXT,
            return REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Portfolios
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Portfolios (
            portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            name TEXT,
            assets TEXT,
            FOREIGN KEY(client_id) REFERENCES Clients(client_id)
        )
    """)

    # Managers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Managers (
            manager_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            portfolio_id INTEGER,
            FOREIGN KEY(portfolio_id) REFERENCES Portfolios(portfolio_id)
        )
    """)

    # Deals
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Deals (
            deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            date TEXT,
            ticker TEXT,
            action TEXT,
            FOREIGN KEY(portfolio_id) REFERENCES Portfolios(portfolio_id)
        )
    """)

    conn.commit()
    conn.close()


# =============================================================================
# PEUPLEMENT DES TABLES
# =============================================================================


# Clients
def insert_clients(db_path="db/Fund.db"):
    
    clients = [
        ("Alice", "Low risk"),
        ("Bob", "Low turnover"),
        ("Charlie", "High yield equity only")
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO Clients (name, risk_profile) VALUES (?, ?)", clients)
    conn.commit()
    conn.close()


# Products
def insert_products(dict_tickers, db_path="db/Fund.db"):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Classification des actifs
    for ticker, name in dict_tickers.items():
        if ticker in ["TLT", "IEF", "AGG", "LQD", "JNK"]:
            asset_class = "Bond"
        elif ticker in ["USO", "UNG", "DBC"]:
            asset_class = "Commodity"
        elif ticker in ["GLD"]:
            asset_class = "Gold"
        elif ticker in ["SPY", "^GSPC"]:
            asset_class = "Benchmark"
        elif ticker in ["^VIX", "^TNX"]:
            asset_class = "Macro"
        else:
            asset_class = "Equity"

        cursor.execute(
            "INSERT OR IGNORE INTO Products (ticker, name, asset_class) VALUES (?, ?, ?)",
            (ticker, name, asset_class)
        )
    conn.commit()
    conn.close()


# Returns
def insert_returns(csv_path="data/processed/prices_clean.csv", db_path="db/Fund.db"):

    # Peuplement de la table à partier du fichier CSV des prix historiques nettoyés
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Calcul des rendements pour les actifs financiers (excluant les indices macroéconomiques)
    macro_cols = ["^VIX", "^TNX"]
    financial_cols = [col for col in df.columns if col not in macro_cols]
    returns_financial = df[financial_cols].pct_change()
    df_returns = pd.concat([returns_financial, df[macro_cols]], axis=1).dropna()
    
    df_long = df_returns.stack().reset_index()
    df_long.columns = ["date", "ticker", "return"]
    
    conn = sqlite3.connect(db_path)
    df_long.to_sql("Returns", conn, if_exists="append", index=False)
    conn.close()


# Portfolios
def insert_portfolios(db_path="db/Fund.db"):

    portfolios = [
        {"client_id": 1, "name": "Portefeuille Low Risk", "assets": []},
        {"client_id": 2, "name": "Portefeuille Low turnover", "assets": []},
        {"client_id": 3, "name": "Portefeuille High yield equity only", "assets": []}
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for pf in portfolios:
        cursor.execute("""
            INSERT INTO Portfolios (client_id, name, assets)
            VALUES (?, ?, ?)
        """, (pf["client_id"], pf["name"], json.dumps(pf["assets"])))

    conn.commit()
    conn.close()


# Managers
def insert_managers(db_path="db/Fund.db"):

    managers = [
        ("Manager A", 1),
        ("Manager B", 2),
        ("Manager C", 3)
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO Managers (name, portfolio_id) VALUES (?, ?)", managers)
    conn.commit()
    conn.close()


# =============================================================================
# FONCTION TERMINALE
# =============================================================================


def build_fund_database(dict_tickers, db_path="db/Fund.db"):
    if os.path.exists(db_path):
        os.remove(db_path)
    create_database(db_path)
    create_tables(db_path)
    insert_clients(db_path)
    insert_products(dict_tickers, db_path)
    insert_returns(db_path=db_path)
    insert_portfolios(db_path)
    insert_managers(db_path)
    print("Base Fund.db créée et entièrement peuplée.")
