# =============================================================================
# INTRODUCTION
# =============================================================================


"""
Module de calcul et d'affichage des principales métriques des portefeuilles multi-asset
Construit un rapport de synthèse avec un classement des managers selon un critère choisi par l'utilisateur, et génère un dashboard graphique des performances.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# CONSTANTES
# =============================================================================


DB_PATH        = "db/Fund.db"
RISK_FREE_RATE = 0.04
TRADING_DAYS   = 252
BENCHMARK      = "SPY"
EVAL_START     = "2023-01-01"
EVAL_END       = "2024-12-31"

RANKING_CRITERIA = {
    "1": ("Rdt ann. (%)",   "Rendement annualise"),
    "2": ("Sharpe",         "Ratio de Sharpe"),
    "3": ("Alpha ann. (%)", "Alpha"),
}

PORTFOLIO_NAMES = {1: "Low Risk", 2: "Low Turnover", 3: "High Yield Equity Only"}
COLORS          = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722"}


# =============================================================================
# CHARGEMENT DES DONNEES DEPUIS LA BASE
# =============================================================================


def load_returns_wide(db_path=DB_PATH):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"""
        SELECT date, ticker, return FROM Returns
        WHERE date >= '{EVAL_START}' AND date <= '{EVAL_END}'
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])

    return df.pivot(index="date", columns="ticker", values="return").sort_index()


def load_deals(portfolio_id, db_path=DB_PATH):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"""
        SELECT * FROM Deals
        WHERE portfolio_id = {portfolio_id}
          AND date >= '{EVAL_START}' AND date <= '{EVAL_END}'
        ORDER BY date
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_initial_assets(portfolio_id, db_path=DB_PATH):

    conn = sqlite3.connect(db_path)
    row = pd.read_sql(f"""
        SELECT assets FROM Portfolios WHERE portfolio_id = {portfolio_id}
    """, conn).iloc[0]["assets"]
    conn.close()

    return json.loads(row)


def load_managers(db_path=DB_PATH):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM Managers", conn)
    conn.close()

    return df


# =============================================================================
# RECONSTRUCTION DE LA NAV
# =============================================================================


def compute_nav(portfolio_id, returns_df, deals_df, initial_assets):
    
    """
    Reconstruit la NAV à partir des données pré-chargées.
    Les DataFrames returns_df, deals_df et initial_assets sont passés en paramètre pour éviter des requêtes SQL redondantes.
    """

    holdings  = set(initial_assets)
    nav_value = 1.0
    nav       = {}

    for date in returns_df.index:
        day_deals = deals_df[deals_df["date"] == date]
        for _, row in day_deals.iterrows():
            if row["action"] == "BUY":
                holdings.add(row["ticker"])
            elif row["action"] == "SELL":
                holdings.discard(row["ticker"])

        available = [t for t in holdings if t in returns_df.columns]
        daily_ret = returns_df.loc[date, available].mean() if available else 0.0
        if pd.isna(daily_ret):
            daily_ret = 0.0

        nav_value *= (1 + daily_ret)
        nav[date]  = nav_value

    return pd.Series(nav)


# =============================================================================
# METRIQUES
# =============================================================================

def annualized_return(nav):

    n_years = len(nav) / TRADING_DAYS

    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1


def annualized_volatility(nav):

    return nav.pct_change().dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(nav):

    ret = annualized_return(nav)
    vol = annualized_volatility(nav)

    return (ret - RISK_FREE_RATE) / vol if vol != 0 else np.nan


def maximum_drawdown(nav):

    return ((nav - nav.cummax()) / nav.cummax()).min()


def beta_alpha(nav, returns_df):

    """
    Calcule le beta et l'alpha annualisés par rapport au benchmark.
    Le DataFrame returns_df est passé en paramètre pour éviter un rechargement SQL à chaque appel.
    """

    if BENCHMARK not in returns_df.columns:
        return np.nan, np.nan

    pf_ret = nav.pct_change().dropna()
    bm_ret = returns_df[BENCHMARK].reindex(pf_ret.index).dropna()
    common = pf_ret.index.intersection(bm_ret.index)

    pf_exc = pf_ret.loc[common] - RISK_FREE_RATE / TRADING_DAYS
    bm_exc = bm_ret.loc[common] - RISK_FREE_RATE / TRADING_DAYS

    cov       = np.cov(pf_exc, bm_exc)
    beta      = cov[0, 1] / cov[1, 1]
    alpha_ann = (pf_exc.mean() - beta * bm_exc.mean()) * TRADING_DAYS

    return beta, alpha_ann


def monthly_turnover(portfolio_id, db_path=DB_PATH):

    deals_df = load_deals(portfolio_id, db_path)
    if deals_df.empty:
        return pd.Series(dtype=float)
    
    return deals_df.set_index("date").resample("ME")["deal_id"].count()


def best_worst_month(nav):

    monthly = nav.resample("ME").last().pct_change().dropna()

    return monthly.max(), monthly.min()


# =============================================================================
# CLASSEMENT DES MANAGERS — CRITERE CHOISI PAR L'UTILISATEUR
# =============================================================================


def ask_ranking_criterion():
    print("\n  Sur quel critere voulez-vous classer les managers ?")
    for key, (_, label) in RANKING_CRITERIA.items():
        print(f"    {key}. {label}")
    choice = input("\n  Votre choix (1-3) : ").strip()
    if choice not in RANKING_CRITERIA:
        print("  Choix invalide, classement par rendement annualise par defaut.")
        choice = "1"

    return RANKING_CRITERIA[choice]


def rank_managers(metrics_df, db_path=DB_PATH):
    col, label = ask_ranking_criterion()

    managers_df = load_managers(db_path)
    merged = managers_df.merge(
        metrics_df[["Nom", "Rdt ann. (%)", "Sharpe", "Alpha ann. (%)"]].reset_index(),
        left_on="portfolio_id", right_on="Portfolio ID", how="left"
    )
    merged = merged[["name", "Nom", "Rdt ann. (%)", "Sharpe", "Alpha ann. (%)"]].copy()
    merged.columns = ["Manager", "Strategie", "Rdt ann. (%)", "Sharpe", "Alpha ann. (%)"]
    merged = merged.sort_values(col, ascending=False).reset_index(drop=True)
    merged.index += 1

    return merged, col, label


# =============================================================================
# GRAPHIQUES
# =============================================================================


def plot_performance(navs, metrics_df):
    names  = [PORTFOLIO_NAMES[pid] for pid in [1, 2, 3]]
    colors = [COLORS[pid] for pid in [1, 2, 3]]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Tableau de bord — Fonds Multi-Asset (2023-2024)",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # 1. NAV
    ax1 = fig.add_subplot(gs[0, :])
    for pid, nav in navs.items():
        ax1.plot(nav.index, nav.values,
                 label=PORTFOLIO_NAMES[pid], color=COLORS[pid], linewidth=1.8)
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("NAV — Valeur cumulee des portefeuilles (base 1)\n"
                  , fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("NAV")

    # 2. Rendement annualisé
    ax2 = fig.add_subplot(gs[1, 0])
    rdts = metrics_df["Rdt ann. (%)"].values
    bars = ax2.bar(names, rdts, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, rdts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Rendement annualise (%)", fontsize=10)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Volatilité
    ax3 = fig.add_subplot(gs[1, 1])
    vols = metrics_df["Vol. ann. (%)"].values
    bars = ax3.bar(names, vols, color=colors, edgecolor="white", linewidth=0.5)
    ax3.axhline(10, color="red", linestyle="--", linewidth=1,
                label="Cible Low Risk (10%)")
    for bar, val in zip(bars, vols):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_title("Volatilite annualisee (%)", fontsize=10)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Sharpe ratio
    ax4 = fig.add_subplot(gs[1, 2])
    sharpes = metrics_df["Sharpe"].values
    bars = ax4.bar(names, sharpes, color=colors, edgecolor="white", linewidth=0.5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.axhline(1, color="green", linestyle="--", linewidth=0.8, label="Sharpe = 1 (bon)")
    for bar, val in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax4.set_title("Ratio de Sharpe\n(rendement / risque)", fontsize=10)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.savefig("performance_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Dashboard sauvegarde -> performance_dashboard.png")


# =============================================================================
# RAPPORT COMPLET
# =============================================================================


def run_performance_report(db_path=DB_PATH):

    print("\n" + "=" * 65)
    print("   RAPPORT DE PERFORMANCE - FONDS MULTI-ASSET (2023-2024)")
    print("=" * 65)

    # Chargement unique de la table Returns — 1 seule requête SQL
    returns_df = load_returns_wide(db_path)

    rows, navs = [], {}

    for pid in [1, 2, 3]:
        # Chargement unique par portfolio (inévitable : pid différent à chaque tour)
        deals_df       = load_deals(pid, db_path)
        initial_assets = load_initial_assets(pid, db_path)

        nav             = compute_nav(pid, returns_df, deals_df, initial_assets)
        navs[pid]       = nav
        b, a            = beta_alpha(nav, returns_df)
        turn            = monthly_turnover(pid, db_path)
        best_m, worst_m = best_worst_month(nav)

        rows.append({
            "Portfolio ID"      : pid,
            "Nom"               : PORTFOLIO_NAMES[pid],
            "Rdt ann. (%)"      : round(annualized_return(nav) * 100, 2),
            "Vol. ann. (%)"     : round(annualized_volatility(nav) * 100, 2),
            "Sharpe"            : round(sharpe_ratio(nav), 3),
            "Max DD (%)"        : round(maximum_drawdown(nav) * 100, 2),
            "Beta"              : round(b, 3) if not np.isnan(b) else "N/A",
            "Alpha ann. (%)"    : round(a * 100, 2) if not np.isnan(a) else "N/A",
            "Meilleur mois (%)" : round(best_m * 100, 2),
            "Pire mois (%)"     : round(worst_m * 100, 2),
            "Deals/mois (moy.)" : round(turn.mean(), 2) if not turn.empty else 0,
        })

    metrics_df = pd.DataFrame(rows).set_index("Portfolio ID")

    # Tableau complet
    print("\nMETRIQUES PAR PORTEFEUILLE\n")
    print(metrics_df.drop(columns=["Nom"]).to_string())

    # Classement des managers
    print("\n" + "-" * 50)
    print("  CLASSEMENT DES MANAGERS")
    print("-" * 50)
    ranking, col, label = rank_managers(metrics_df, db_path)
    print(f"\n  Classement par : {label}\n")
    print(ranking.to_string())
    best = ranking.iloc[0]
    print(f"\n  Meilleur manager : {best['Manager']} "
          f"({best['Strategie']}) - {label} = {best[col]:.3f}")

    # Graphiques
    print("\n  Generation du dashboard...")
    plot_performance(navs, metrics_df)

    print("\n" + "=" * 65)
    print("  Rapport termine.")
    print("=" * 65 + "\n")