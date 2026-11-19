# Crypto SEPA+VCP Screener

Automated daily screener for the largest cryptocurrencies using Minervini's SEPA trend template, VCP scoring, stage classification and the shared rule-based decision engine. Crypto sibling of [asx-sepa-vcp-screener](https://github.com/KidLearning99/asx-sepa-vcp-screener), [us-sepa-vcp-screener](https://github.com/KidLearning99/us-sepa-vcp-screener) and [india-sepa-vcp-screener](https://github.com/KidLearning99/india-sepa-vcp-screener).

- **Universe:** the top 200 coins by market cap from CoinGecko that trade on a supported exchange. Stablecoins, wrapped, bridged and liquid-staking tokens, and tokenized gold, stocks and treasuries are left out (CoinGecko categories, a symbol list, and a check that drops anything with under 10% annualised volatility). Change the size with the `UNIVERSE_SIZE` env var.
- **Daily candles:** Binance spot (USDT pairs), then Coinbase, Kraken (USD) and Gate (USDT) for coins Binance does not list. Only completed UTC days are used. A coin whose exchange price is more than 25% away from CoinGecko's is treated as a different coin with the same ticker and skipped.
- **Benchmark:** a top-10 coin index (the ten largest eligible coins, weighted by today's market cap) for relative strength, beta, correlation and the market filter.
- **Crypto settings:** a year is 365 days. Coins have no revenue, earnings or analyst coverage, so the fundamentals leg is a manual check, and the dashboard shows token supply instead: circulating share of supply, market cap vs fully diluted value, and distance from the all-time high.
- **Categories:** `data/categories.json` is a snapshot of CoinGecko's category lists, refreshed weekly by the workflow (CoinGecko's free API allows only a few calls a minute). An optional `COINGECKO_API_KEY` repository secret (free demo key) raises that limit.
- **Schedule:** GitHub Actions, daily at 00:25 UTC after the daily candle closes. Output is deployed to the `gh-pages` branch.

`decision.py` is shared with the share screeners; crypto settings and wording are set at the top of `screener.py`.

Educational and research use only. Not financial advice. Prices in USD.
