import argparse
import os
import json
import math
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

np.random.seed(42)
random.seed(42)


ARTIFACT_DIR = "./artifacts"


def ensure_artifacts_dir():
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)


def render_equity_curve(equity_curve, show_plot=False, save_plot=True):
    ensure_artifacts_dir()
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    plot_path = None
    if save_plot:
        plot_path = os.path.join(ARTIFACT_DIR, 'equity_curve.png')
        plt.savefig(plot_path)
    if show_plot:
        plt.show()
    else:
        plt.close()
    return plot_path


def write_summary_artifact(payload):
    ensure_artifacts_dir()
    with open(os.path.join(ARTIFACT_DIR, 'summary.txt'), 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


def generate_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start_date, end=end_date)


def generate_data():
    print("Generating synthetic data...")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365 * 3)
    dates = generate_calendar(start_date, end_date)
    n = len(dates)
    tickers = [f"T{i}" for i in range(20)]

    # Generate correlated returns using random covariance matrix
    base_cov = np.random.randn(20, 20)
    cov_matrix = np.dot(base_cov, base_cov.T)
    cov_matrix /= np.max(np.abs(cov_matrix))
    cov_matrix += np.eye(20) * 0.05
    mean_returns = np.linspace(0.0001, 0.0005, 20)
    daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix * 0.0004, size=n)

    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    prices.iloc[0] = 100 * (1 + np.random.randn(20) * 0.01)
    for t in range(1, n):
        prices.iloc[t] = prices.iloc[t - 1] * (1 + daily_returns[t])
    prices = prices.abs()

    returns = prices.pct_change().fillna(0.0)

    # Quarterly fundamentals
    q_dates = pd.date_range(start=dates[0] - pd.Timedelta(days=30), end=dates[-1], freq='Q')
    fundamentals = []
    pe_level = np.random.uniform(10, 25, size=20)
    pb_level = np.random.uniform(1, 5, size=20)
    margin_level = np.random.uniform(0.1, 0.3, size=20)
    eps_level = np.random.uniform(1, 3, size=20)
    for qd in q_dates:
        pe_level = pe_level * 0.9 + np.random.uniform(10, 25, size=20) * 0.1 + np.random.randn(20) * 0.2
        pb_level = pb_level * 0.9 + np.random.uniform(1, 5, size=20) * 0.1 + np.random.randn(20) * 0.1
        margin_level = margin_level * 0.85 + np.random.uniform(0.05, 0.35, size=20) * 0.15 + np.random.randn(20) * 0.01
        eps_level = eps_level * 0.85 + np.random.uniform(1, 4, size=20) * 0.15 + np.random.randn(20) * 0.05
        fundamentals.append(pd.DataFrame({
            'date': qd,
            'ticker': tickers,
            'pe': pe_level,
            'pb': pb_level,
            'margin': margin_level,
            'eps': eps_level
        }))
    fundamentals = pd.concat(fundamentals, ignore_index=True)

    # Weekly analyst revisions correlated with returns
    weekly_dates = pd.date_range(start=dates[0], end=dates[-1], freq='W')
    analyst_records = []
    for wd in weekly_dates:
        base_revision = np.random.randn(20) * 0.05
        linked_returns = returns.loc[returns.index <= wd].tail(5).mean().values
        revisions = base_revision + 0.5 * np.nan_to_num(linked_returns)
        ratings = 3 + np.random.randn(20) * 0.2
        analyst_records.append(pd.DataFrame({
            'date': wd,
            'ticker': tickers,
            'revision': revisions,
            'rating': ratings
        }))
    analyst = pd.concat(analyst_records, ignore_index=True)

    # Macro features
    macro = pd.DataFrame(index=dates)
    macro['vix'] = 20 + np.cumsum(np.random.randn(n) * 0.1)
    macro['yield_spread'] = 1 + np.cumsum(np.random.randn(n) * 0.01)
    macro['pmi'] = 50 + np.cumsum(np.random.randn(n) * 0.05)
    macro['oil'] = 60 + np.cumsum(np.random.randn(n) * 0.3)

    # Hidden regime assignment
    bull = (macro['pmi'] > 52) & (macro['vix'] < 18) & (macro['yield_spread'] > 1)
    bear = (macro['pmi'] < 48) | (macro['vix'] > 25)
    regime = np.select([bull, bear], [0, 2], default=1)
    macro['regime'] = regime

    return {
        'dates': dates,
        'tickers': tickers,
        'prices': prices,
        'returns': returns,
        'fundamentals': fundamentals,
        'analyst': analyst,
        'macro': macro
    }


def compute_factors(data):
    print("Computing factors...")
    prices = data['prices']
    returns = data['returns']
    fundamentals = data['fundamentals']
    analyst = data['analyst']
    macro = data['macro']
    tickers = data['tickers']

    # Prepare fundamentals to daily
    fund_pivot = fundamentals.pivot(index='date', columns='ticker', values=['pe', 'margin', 'eps'])
    fund_pivot = fund_pivot.sort_index().reindex(prices.index, method='ffill')

    pe = fund_pivot['pe']
    margin = fund_pivot['margin']

    margin_trend = margin.diff().rolling(63).mean().fillna(0)
    pe_z = ((pe - pe.rolling(252, min_periods=20).mean()) / pe.rolling(252, min_periods=20).std()).fillna(0)
    valuation = (-pe_z + margin_trend).fillna(0)

    # Momentum
    def total_return(window):
        return prices.pct_change(window).replace([np.inf, -np.inf], np.nan)

    mom_1m = total_return(21)
    mom_3m = total_return(63)
    mom_6m = total_return(126)
    momentum = (mom_1m.rank(axis=1, pct=True) + mom_3m.rank(axis=1, pct=True) + mom_6m.rank(axis=1, pct=True)) / 3

    # Analyst data to daily
    analyst_pivot = analyst.pivot(index='date', columns='ticker', values=['revision', 'rating']).sort_index()
    analyst_pivot = analyst_pivot.reindex(prices.index, method='ffill')
    revision = analyst_pivot['revision'].fillna(0)
    rating = analyst_pivot['rating']
    rating = rating.fillna(rating.mean())
    revision_trend = revision.rolling(4).mean().fillna(0)
    analyst_factor = revision_trend + (rating - rating.mean())

    # Macro/Industry: Beta to sector index (mean returns) and regime dummy
    sector_index = prices.mean(axis=1)
    sector_ret = sector_index.pct_change().fillna(0)
    rolling_corr = returns.rolling(63).corr(sector_ret)
    asset_std = returns.rolling(63).std()
    sector_std = sector_ret.rolling(63).std().replace(0, np.nan)
    beta = (rolling_corr * (asset_std.div(sector_std, axis=0))).fillna(0)
    regime_dummy = pd.get_dummies(macro['regime']).reindex(prices.index).fillna(method='ffill').fillna(0)
    if 0 not in regime_dummy.columns:
        regime_dummy[0] = 0
    if 1 not in regime_dummy.columns:
        regime_dummy[1] = 0
    if 2 not in regime_dummy.columns:
        regime_dummy[2] = 0
    regime_dummy = regime_dummy[[0, 1, 2]]
    bull_boost = regime_dummy[0].values.reshape(-1, 1)
    bear_penalty = regime_dummy[2].values.reshape(-1, 1)
    macro_factor = beta.values + bull_boost - bear_penalty
    macro_factor = pd.DataFrame(macro_factor, index=prices.index, columns=prices.columns)

    # Volatility factor
    realized_vol = returns.rolling(20).std().replace(0, np.nan)
    volatility_factor = (1 / realized_vol).replace([np.inf, -np.inf], 0).fillna(0)

    # Standardize cross-sectionally for each date
    factor_dict = {}
    factors = {
        'valuation': valuation,
        'momentum': momentum,
        'analyst': analyst_factor,
        'macro': macro_factor,
        'volatility': volatility_factor
    }
    for name, df in factors.items():
        z = df.apply(lambda row: (row - row.mean()) / (row.std() + 1e-6), axis=1)
        factor_dict[name] = z.fillna(0)

    return factor_dict


def train_regime_model(data):
    print("Training regime classifier...")
    macro = data['macro']
    features = macro[['vix', 'yield_spread', 'pmi', 'oil']]
    target = macro['regime']
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = LogisticRegression(multi_class='multinomial', max_iter=500)
    model.fit(X, target)
    probs = model.predict_proba(X)
    predicted = model.predict(X)
    prob_df = pd.DataFrame(probs, index=features.index, columns=model.classes_)
    prob_df = prob_df.reindex(columns=[0, 1, 2], fill_value=0)
    regime_series = pd.Series(predicted, index=features.index)
    return model, scaler, prob_df, regime_series


def get_regime_weights(prob_row):
    bull_w = np.array([0.20, 0.35, 0.20, 0.15, 0.10])  # Valuation, Momentum, Analyst, Macro, Vol
    neutral_w = np.array([0.22, 0.22, 0.20, 0.18, 0.18])
    bear_w = np.array([0.30, 0.15, 0.15, 0.15, 0.25])
    total_prob = prob_row.get(0, 0) + prob_row.get(1, 0) + prob_row.get(2, 0)
    if total_prob == 0:
        weights = neutral_w.copy()
    else:
        weights = (prob_row.get(0, 0) * bull_w + prob_row.get(1, 0) * neutral_w + prob_row.get(2, 0) * bear_w)
    weights /= weights.sum()
    return weights


def score_tickers(data, factors, prob_df, regime_series):
    print("Scoring tickers...")
    dates = data['prices'].index
    tickers = data['tickers']

    factor_names = ['valuation', 'momentum', 'analyst', 'macro', 'volatility']
    rows = []
    for date in dates:
        prob_row_list = prob_df.reindex([date]).fillna(0).to_dict(orient='records')
        prob_row = prob_row_list[0] if prob_row_list else {}
        weights = get_regime_weights(prob_row)
        factor_values = [factors[name].loc[date] for name in factor_names]
        for idx, ticker in enumerate(tickers):
            factor_vector = np.array([f.loc[ticker] for f in factor_values])
            composite = np.dot(weights, factor_vector)
            score = 100 * (1 / (1 + math.exp(-composite)))
            score = min(max(score, 1), 100)
            rows.append({
                'date': date,
                'ticker': ticker,
                'score': score,
                'regime': regime_series.loc[date]
            })
    scores = pd.DataFrame(rows)
    return scores


def allocate(data, scores):
    print("Allocating portfolio weights...")
    returns = data['returns']
    prices = data['prices']
    tickers = data['tickers']

    score_pivot = scores.pivot(index='date', columns='ticker', values='score')
    mu = (score_pivot - 50) / 500

    vol = returns.rolling(20).std().reindex(score_pivot.index).fillna(method='bfill').replace(0, np.nan)
    vol = vol.fillna(vol.median())

    alpha = 0.3
    rebalance_dates = score_pivot.index[::5]
    weights = pd.DataFrame(0, index=score_pivot.index, columns=tickers, dtype=float)

    prev_weights = pd.Series(1 / len(tickers), index=tickers)
    for date in score_pivot.index:
        if date in rebalance_dates:
            mu_row = mu.loc[date]
            vol_row = vol.loc[date]
            raw = mu_row / (vol_row + 1e-6)
            raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
            if raw.abs().sum() == 0:
                raw = pd.Series(0, index=tickers)
            else:
                raw = raw / raw.abs().sum()
            eq = pd.Series(1 / len(tickers), index=tickers)
            new_weights = alpha * raw + (1 - alpha) * eq
            prev_weights = new_weights
        weights.loc[date] = prev_weights

    portfolio_returns = (weights.shift().fillna(1 / len(tickers)) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).mean()

    return weights, portfolio_returns, turnover


def calculate_metrics(portfolio_returns):
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()
    sharpe = (daily_mean / (daily_std + 1e-6)) * np.sqrt(252)
    downside = portfolio_returns[portfolio_returns < 0]
    sortino = (daily_mean / (downside.std() + 1e-6)) * np.sqrt(252)
    equity_curve = (1 + portfolio_returns).cumprod()
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1
    max_dd = drawdown.min()
    annual_return = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'annualized_return': float(annual_return)
    }, equity_curve


def backtest(data, scores, weights, portfolio_returns, turnover):
    print("Running backtest...")
    metrics, equity_curve = calculate_metrics(portfolio_returns)
    metrics['turnover'] = float(turnover)
    metrics['exposure_long'] = float(weights.clip(lower=0).sum(axis=1).mean())
    metrics['exposure_short'] = float(weights.clip(upper=0).sum(axis=1).mean())
    return metrics, equity_curve


def summarize_latest_state(scores, weights):
    latest_date = scores['date'].max()
    latest_scores = scores[scores['date'] == latest_date].copy()
    latest_weights = weights.loc[latest_date]
    latest_scores['weight'] = latest_scores['ticker'].map(latest_weights)
    latest_scores = latest_scores.sort_values('score', ascending=False)
    current_regime = latest_scores['regime'].iloc[0]
    top5 = latest_scores.head(5)['ticker'].tolist()
    return latest_date, latest_scores, current_regime, top5


def make_report(
    data,
    scores,
    weights,
    metrics,
    equity_curve,
    plot_path=None,
    export_pdf=True,
):
    print("Generating reports...")
    ensure_artifacts_dir()

    latest_date, latest_scores, current_regime, top5 = summarize_latest_state(scores, weights)
    latest_scores.to_csv(os.path.join(ARTIFACT_DIR, 'bsh3_scores.csv'), index=False)

    regime_map = {0: 'bull', 1: 'neutral', 2: 'bear'}
    top5_df = latest_scores.head(5)

    if export_pdf:
        if not plot_path:
            plot_path = render_equity_curve(equity_curve, show_plot=False, save_plot=True)
        pdf_path = os.path.join(ARTIFACT_DIR, 'bsh3_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "BSH 3.0 Auto-Scorer Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Current Regime: {regime_map.get(current_regime, 'unknown')}")

        c.drawString(50, height - 110, "Top 5 Tickers:")
        y = height - 130
        for _, row in top5_df.iterrows():
            c.drawString(60, y, f"{row['ticker']}: Score {row['score']:.1f}, Weight {row['weight']:.3f}")
            y -= 15

        c.drawString(50, y - 10, f"Sharpe: {metrics['sharpe']:.2f}")
        c.drawString(200, y - 10, f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        c.drawString(50, y - 30, f"Annualized Return: {metrics['annualized_return']:.2%}")
        c.drawString(200, y - 30, f"Turnover: {metrics['turnover']:.2f}")

        if plot_path and os.path.exists(plot_path):
            c.drawImage(plot_path, 50, 100, width=500, height=250)
        c.showPage()
        c.save()

    return latest_scores, regime_map.get(current_regime, 'unknown'), top5


def run_pipeline(show_plot=True, save_plot=True, export_pdf=True, quiet=False):
    ensure_artifacts_dir()
    data = generate_data()
    factors = compute_factors(data)
    model, scaler, prob_df, regime_series = train_regime_model(data)
    scores = score_tickers(data, factors, prob_df, regime_series)
    weights, portfolio_returns, turnover = allocate(data, scores)
    metrics, equity_curve = backtest(data, scores, weights, portfolio_returns, turnover)

    combined_plot = save_plot or export_pdf or show_plot
    plot_path = None
    if combined_plot:
        plot_path = render_equity_curve(
            equity_curve,
            show_plot=show_plot,
            save_plot=save_plot or export_pdf,
        )

    latest_scores, final_regime, top5 = make_report(
        data,
        scores,
        weights,
        metrics,
        equity_curve,
        plot_path=plot_path,
        export_pdf=export_pdf,
    )

    summary = {
        'final_regime': final_regime,
        'annualized_return': round(metrics['annualized_return'], 4),
        'sharpe': round(metrics['sharpe'], 2),
        'max_drawdown': round(metrics['max_drawdown'], 4),
        'top5': top5,
    }
    summary_payload = dict(metrics)
    summary_payload.update(summary)
    write_summary_artifact(summary_payload)
    if not quiet:
        print(summary)

    return {
        'summary': summary,
        'metrics': metrics,
        'scores': latest_scores,
        'weights': weights,
        'equity_curve': equity_curve,
        'plot_path': plot_path,
    }


def load_latest_summary():
    summary_path = os.path.join(ARTIFACT_DIR, 'summary.txt')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Warning: summary.txt is not valid JSON.")
    return None


def load_latest_scores():
    scores_path = os.path.join(ARTIFACT_DIR, 'bsh3_scores.csv')
    if os.path.exists(scores_path):
        try:
            return pd.read_csv(scores_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: unable to read scores CSV ({exc}).")
    return None


def inspect_artifacts(top=5, show_weights=False):
    ensure_artifacts_dir()
    summary = load_latest_summary()
    if summary:
        print("Latest summary metrics:")
        print(json.dumps(summary, indent=2))
    else:
        print("No summary metrics found. Run the pipeline first.")

    scores = load_latest_scores()
    if scores is not None and not scores.empty:
        display_cols = ['ticker', 'score']
        if show_weights and 'weight' in scores.columns:
            display_cols.append('weight')
        print(f"Top {top} scores from latest run:")
        print(scores[display_cols].head(top).to_string(index=False))
    else:
        print("No score artifacts found. Run the pipeline to generate them.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Regime-aware BSH 3.0 Auto-Scorer simulation platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help="Execute the full pipeline")
    run_parser.add_argument(
        '--no-show-plot',
        action='store_true',
        help="Skip displaying the equity curve plot interactively.",
    )
    run_parser.add_argument(
        '--no-save-plot',
        action='store_true',
        help="Do not save the equity curve PNG artifact.",
    )
    run_parser.add_argument(
        '--no-pdf',
        action='store_true',
        help="Skip generating the PDF report (other artifacts still save).",
    )
    run_parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress the final JSON-style summary printout.",
    )

    inspect_parser = subparsers.add_parser(
        'inspect',
        help="Inspect the most recent artifacts without re-running the simulation",
    )
    inspect_parser.add_argument('--top', type=int, default=5, help="Number of top tickers to display.")
    inspect_parser.add_argument(
        '--show-weights',
        action='store_true',
        help="Display portfolio weights alongside scores when available.",
    )

    parser.set_defaults(
        command='run',
        no_show_plot=False,
        no_save_plot=False,
        no_pdf=False,
        quiet=False,
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == 'inspect':
        inspect_artifacts(top=args.top, show_weights=args.show_weights)
        return

    if args.command == 'run':
        run_pipeline(
            show_plot=not args.no_show_plot,
            save_plot=not args.no_save_plot,
            export_pdf=not args.no_pdf,
            quiet=args.quiet,
        )
        return

    parser.print_help()


if __name__ == '__main__':
    main()
