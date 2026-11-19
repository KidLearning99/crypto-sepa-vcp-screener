"""
Crypto SEPA+VCP screener.

Crypto sibling of the ASX, US and India SEPA+VCP screeners. Same trend template,
VCP scoring, stage classification and shared decision engine (decision.py),
adapted for coins:

- Universe: the top UNIVERSE_SIZE coins by market cap (CoinGecko) that trade on a
  supported exchange, after removing stablecoins, wrapped, bridged and staked
  tokens, and tokenized gold, stocks and treasuries.
- Daily candles: Binance spot (USDT), then Coinbase (USD), Kraken (USD) and
  Gate (USDT) for coins Binance does not list. Only completed UTC days are used.
- Benchmark: a top-10 coin index (the ten largest eligible coins, weighted by
  today's market cap). Relative strength, the market filter and beta use it.
- A year is 365 days (crypto trades every day), not 252 sessions.

Educational and research use only. Not financial advice. Prices in USD.
"""
import decision
import json, os, re, sys, time, math, threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

MELBOURNE = ZoneInfo("Australia/Melbourne")

UNIVERSE_SIZE   = int(os.environ.get("UNIVERSE_SIZE", 200))
SEPA_MIN        = 3
MIN_ADV         = 1_000_000   # thin-trading threshold: avg daily traded value (USD) on the exchange used
VOL_BREAKOUT    = 1.5
VCP_MIN         = 2
YEAR            = 365         # days in a year - crypto trades every day
HISTORY_DAYS    = 1000        # ~2.7 years: 1 year for scoring, the rest for the backtest
BENCH_N         = 10          # coins in the benchmark index
PEG_VOL_MAX     = 10.0        # 90-day annualised volatility (%) below this = pegged asset, skipped
PRICE_TOL       = 0.25        # exchange close must be within 25% of CoinGecko's price (catches ticker clashes)
CATEGORY_CACHE  = "data/categories.json"
CATEGORY_MAX_AGE = 7           # days before the category snapshot is refreshed

# ── Shared decision engine: crypto settings and wording ───────────────────────
decision.YEAR          = YEAR
decision.T2_FLOOR      = 0.30   # keep the second target above target 1 (+20%)
decision.FUND_NA_HEAD  = "No revenue or earnings for a coin"
decision.FUND_NA_TEXT  = ("Coins have no revenue or earnings to test, so this leg cannot be scored. "
                          "Check the token's supply schedule (upcoming unlocks, inflation) and the project's "
                          "news yourself before buying.")
decision.TREND_BEAT    = "the coin is beating the top-10 coin index"
decision.CATALYST_TEXT = ("A real catalyst (a network upgrade, a major exchange listing, an ETF or partnership, "
                          "or a large token unlock ending) has to be found in the project's own announcements.")

BENCH_NAME = "Top-10 coin index"

# ── What is not a "coin" for this screener ────────────────────────────────────
EXCLUDE_CATEGORIES = [
    "stablecoins", "yield-bearing-stablecoins", "bridged-stablecoins",
    "wrapped-tokens", "bridged-tokens",
    "liquid-staking-tokens", "liquid-restaking-tokens",
    "tokenized-gold", "tokenized-commodities", "tokenized-treasuries", "tokenized-stock",
]
# Backstop when CoinGecko's category lists cannot be loaded
EXCLUDE_SYMBOLS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USD1", "RLUSD", "USDD", "FRAX",
    "GUSD", "USDP", "BUSD", "LUSD", "CRVUSD", "GHO", "USD0", "USDX", "EURC", "EURT", "EURS", "XAUT",
    "PAXG", "KAU", "KAG", "WBTC", "WETH", "CBBTC", "WEETH", "STETH", "WSTETH", "RETH", "METH", "EZETH",
    "RSETH", "BNSOL", "JITOSOL", "MSOL", "JUPSOL", "SUSDE", "SUSDS", "SDAI", "USDY", "BUIDL", "USYC",
    "OUSG", "USTB", "BFUSD", "USDF", "USDAI", "SOLVBTC", "LBTC", "TBTC", "WBETH", "CMETH", "WBNB",
}
EXCLUDE_SYMBOL_RE = re.compile(r"^(USD|EUR)|USD$|USD\d")
EXCLUDE_NAME_RE = re.compile(r"\b(wrapped|bridged|staked|restaked|liquid staking|stablecoin|tokenized|"
                             r"xstock|treasury|t-bills?|money market|overnight swap|clo fund)\b", re.I)

# Sector label = first of these CoinGecko categories the coin belongs to
SECTOR_CATEGORIES = [
    ("centralized-exchange-token-cex", "Exchange token"), ("exchange-based-tokens", "Exchange token"),
    ("meme-token", "Meme"), ("artificial-intelligence", "AI"), ("real-world-assets-rwa", "RWA"),
    ("gaming", "Gaming"), ("privacy", "Privacy"), ("oracle", "Oracle"), ("depin", "DePIN"),
    ("storage", "Storage"), ("decentralized-exchange", "DEX"), ("decentralized-finance-defi", "DeFi"),
    ("layer-2", "Layer 2"), ("layer-1", "Layer 1"), ("payment-solutions", "Payments"),
    ("infrastructure", "Infrastructure"),
]


def now_melb():
    return datetime.now(MELBOURNE)


def fmt_cap(c):
    if c >= 1e12: return f"${c/1e12:.2f}T"
    if c >= 1e9:  return f"${c/1e9:.1f}B"
    if c >= 1e6:  return f"${c/1e6:.0f}M"
    return f"${c:.0f}"


def fmt_pct(v):
    return ('+' if v >= 0 else '') + str(round(v, 2)) + '%'


def fmt_price(v):
    return "$" + decision._ps(v)


# ── HTTP with a per-host pace limit ───────────────────────────────────────────
_session = requests.Session()
_session.headers["User-Agent"] = "crypto-sepa-vcp-screener (+https://github.com/KidLearning99/crypto-sepa-vcp-screener)"
_pace_lock = threading.Lock()
_pace_last = {}
PACE = {"api.coingecko.com": 2.2 if os.environ.get("COINGECKO_API_KEY") else 15.0, "data-api.binance.vision": 0.05, "api.exchange.coinbase.com": 0.15,
        "api.kraken.com": 1.1, "api.gateio.ws": 0.12}


def http_json(url, params=None, headers=None, tries=4):
    host = url.split("/")[2]
    err = None
    for attempt in range(tries):
        with _pace_lock:
            wait = PACE.get(host, 0.1) - (time.time() - _pace_last.get(host, 0))
            _pace_last[host] = time.time() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            err = f"{type(e).__name__}"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            err = f"HTTP {r.status_code}"
            time.sleep(min(20 * (attempt + 1), 65) if "coingecko" in host else 3 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise ValueError(f"HTTP {r.status_code}")
        return r.json()
    raise RuntimeError(f"{host} failed after {tries} tries: {err}")


CG = "https://api.coingecko.com/api/v3"
CG_HEADERS = {"x-cg-demo-api-key": os.environ["COINGECKO_API_KEY"]} if os.environ.get("COINGECKO_API_KEY") else None


def cg(path, **params):
    return http_json(CG + path, params=params, headers=CG_HEADERS, tries=6)


# ── Universe ──────────────────────────────────────────────────────────────────
def load_markets(pages=2):
    rows = []
    for p in range(1, pages + 1):
        rows += cg("/coins/markets", vs_currency="usd", order="market_cap_desc", per_page=250, page=p,
                   sparkline="false", price_change_percentage="7d,30d") or []
    rows = [r for r in rows if r.get("market_cap") and r.get("current_price")]
    if len(rows) < 100:
        raise RuntimeError(f"CoinGecko returned only {len(rows)} coins - aborting rather than "
                           "publishing a thin dashboard.")
    return rows


def load_categories():
    """category id -> set of coin ids, for the exclusion and sector categories.

    CoinGecko's free API allows only a few calls a minute, so the lists live in
    a committed snapshot (data/categories.json) that is refreshed when it is
    over CATEGORY_MAX_AGE days old. Any list that fails to refresh keeps its
    snapshot copy; the workflow commits the refreshed file back to main."""
    try:
        with open(CATEGORY_CACHE, encoding="utf-8") as fh:
            snap = json.load(fh) or {}
    except Exception:
        snap = {}
    cache = snap.get("categories", {})
    wanted = list(dict.fromkeys(EXCLUDE_CATEGORIES + [c for c, _ in SECTOR_CATEGORIES]))
    try:
        age = (now_melb().date() - datetime.strptime(snap.get("updated", ""), "%Y-%m-%d").date()).days
    except Exception:
        age = 10**6
    if age <= CATEGORY_MAX_AGE and all(c in cache for c in wanted):
        print(f"Categories: snapshot from {snap['updated']} ({age} days old)")
        return {k: set(v) for k, v in cache.items()}
    out, fresh = {}, 0
    for cid in wanted:
        try:
            rows = cg("/coins/markets", vs_currency="usd", category=cid, order="market_cap_desc", per_page=250, page=1)
            out[cid] = sorted({r["id"] for r in rows})
            fresh += 1
        except Exception as e:
            out[cid] = cache.get(cid, [])
            print(f"  category {cid}: {e} - using snapshot ({len(out[cid])} coins)")
    print(f"Categories: {fresh} refreshed, {len(wanted) - fresh} from snapshot")
    if fresh == len(wanted) or (fresh and not cache):
        with open(CATEGORY_CACHE, "w", encoding="utf-8") as fh:
            json.dump({"about": "CoinGecko category members used to exclude stablecoins, wrapped/staked tokens and "
                                "tokenized assets, and to label each coin's sector. Refreshed weekly by the workflow.",
                       "updated": now_melb().strftime("%Y-%m-%d"), "categories": out}, fh, indent=1)
            fh.write("\n")
    return {k: set(v) for k, v in out.items()}


def excluded_reason(c, cats):
    sym, name = c["symbol"].upper(), c.get("name") or ""
    for cid in EXCLUDE_CATEGORIES:
        if c["id"] in cats.get(cid, ()):
            return cid
    if sym in EXCLUDE_SYMBOLS or EXCLUDE_SYMBOL_RE.search(sym):
        return "stable/wrapped symbol"
    if EXCLUDE_NAME_RE.search(name):
        return "wrapped/staked/tokenized name"
    return None


def sector_of(cid, cats):
    for cat, label in SECTOR_CATEGORIES:
        if cid in cats.get(cat, ()):
            return label
    return "Other"


# ── Exchanges: daily candles ──────────────────────────────────────────────────
class Exchanges:
    """Lists which base assets each exchange quotes in USD/USDT, then fetches
    completed daily candles as a DataFrame (Open, High, Low, Close, Volume)
    indexed by UTC date. Volume is in coin units, so Close x Volume = USD."""

    def __init__(self):
        self.lists = {}
        loaders = (("Binance", self._list_binance), ("Coinbase", self._list_coinbase),
                   ("Kraken", self._list_kraken), ("Gate", self._list_gate))
        for name, fn in loaders:
            try:
                self.lists[name] = fn()
                print(f"  {name}: {len(self.lists[name])} USD/USDT markets")
            except Exception as e:
                self.lists[name] = {}
                print(f"  {name}: market list unavailable ({e})")
        if not any(self.lists.values()):
            raise RuntimeError("No exchange market lists loaded - aborting.")
        self.today = pd.Timestamp(datetime.now(timezone.utc).date())

    @staticmethod
    def _list_binance():
        info = http_json("https://data-api.binance.vision/api/v3/exchangeInfo", {"permissions": "SPOT"})
        return {s["baseAsset"].upper(): s["symbol"] for s in info["symbols"]
                if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}

    @staticmethod
    def _list_coinbase():
        rows = http_json("https://api.exchange.coinbase.com/products")
        return {p["base_currency"].upper(): p["id"] for p in rows
                if p["quote_currency"] == "USD" and p.get("status") == "online" and not p.get("trading_disabled")}

    @staticmethod
    def _list_kraken():
        res = http_json("https://api.kraken.com/0/public/AssetPairs")["result"]
        out = {}
        for key, v in res.items():
            ws = v.get("wsname") or ""
            if ws.endswith("/USD") and not key.endswith(".d"):
                base = ws.split("/")[0].upper()
                out[{"XBT": "BTC", "XDG": "DOGE"}.get(base, base)] = key
        return out

    @staticmethod
    def _list_gate():
        rows = http_json("https://api.gateio.ws/api/v4/spot/currency_pairs")
        return {p["base"].upper(): p["id"] for p in rows
                if p.get("quote") == "USDT" and p.get("trade_status") == "tradable"}

    def _frame(self, rows):
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], unit="s").dt.normalize()
        df = df.astype({k: float for k in ("Open", "High", "Low", "Close", "Volume")})
        df = df.drop_duplicates("Date").set_index("Date").sort_index()
        df = df[df.index < self.today]                       # completed UTC days only
        return df[(df["Close"] > 0) & (df["High"] >= df["Low"])]

    def binance(self, sym):
        k = http_json("https://data-api.binance.vision/api/v3/klines",
                      {"symbol": sym, "interval": "1d", "limit": HISTORY_DAYS})
        return self._frame([[x[0] // 1000, x[1], x[2], x[3], x[4], x[5]] for x in k])

    def coinbase(self, pid):
        rows, end = [], int(time.time())
        for _ in range(4):                                   # 300 candles per request
            start = end - 300 * 86400
            chunk = http_json(f"https://api.exchange.coinbase.com/products/{pid}/candles",
                              {"granularity": 86400, "start": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                               "end": datetime.fromtimestamp(end, timezone.utc).isoformat()})
            if not chunk:
                break
            rows += [[x[0], x[3], x[2], x[1], x[4], x[5]] for x in chunk]
            end = start
        return self._frame(rows).tail(HISTORY_DAYS)

    def kraken(self, pair):
        res = http_json("https://api.kraken.com/0/public/OHLC", {"pair": pair, "interval": 1440})["result"]
        key = next(k for k in res if k != "last")
        return self._frame([[x[0], x[1], x[2], x[3], x[4], x[6]] for x in res[key]])

    def gate(self, pair):
        k = http_json("https://api.gateio.ws/api/v4/spot/candlesticks",
                      {"currency_pair": pair, "interval": "1d", "limit": HISTORY_DAYS})
        return self._frame([[int(x[0]), x[5], x[3], x[4], x[2], x[6]] for x in k])

    def history(self, symbol, cg_price):
        """Best candle history for a coin: the first exchange with a sane price
        and at least ~13 months of data, else the longest sane one found."""
        best = None
        for name, fn in (("Binance", self.binance), ("Coinbase", self.coinbase),
                         ("Kraken", self.kraken), ("Gate", self.gate)):
            market = self.lists.get(name, {}).get(symbol)
            if not market:
                continue
            try:
                df = fn(market)
            except Exception:
                continue
            if len(df) < 30:
                continue
            last = float(df["Close"].iloc[-1])
            if not cg_price or abs(last / cg_price - 1) > PRICE_TOL:
                continue                                     # different coin with the same ticker, or stale market
            if best is None or len(df) > len(best[0]):
                best = (df, name, market)
            if len(df) >= YEAR + 30:
                break
        return best


# ── Benchmark: top-10 coin index ──────────────────────────────────────────────
def build_index(coins, hist_map):
    """Daily level of the ten largest eligible coins, weighted by today's market
    cap. On days a member has no candle the weights of the others are rescaled."""
    members = [c for c in coins if c["key"] in hist_map][:BENCH_N]
    if not members:
        return None, []
    closes = pd.DataFrame({c["key"]: hist_map[c["key"]]["Close"] for c in members}).sort_index()
    rets = closes.pct_change()
    w = pd.Series({c["key"]: float(c["market_cap"]) for c in members})
    have = rets.notna()
    wsum = have.mul(w, axis=1).sum(axis=1)
    idx_ret = (rets.fillna(0).mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)).fillna(0)
    level = 1000 * (1 + idx_ret).cumprod()
    return level.rename(BENCH_NAME), [c["symbol"].upper() for c in members]


# ── Technical indicators (same set as the share screeners, 365-day year) ──────
BENCH_CLOSE = None


def technicals(hist):
    out = {}

    def _r(v, d=2):
        try:
            v = float(v)
            return round(v, d) if v == v and abs(v) != float("inf") else None
        except Exception:
            return None

    def _rp(v):            # prices keep significant figures for sub-cent coins
        return decision._p(v)
    try:
        c = hist["Close"].astype(float)
        h = hist["High"].astype(float)
        l = hist["Low"].astype(float)
        px = c.iloc[-1]

        if len(c) > 15:
            d = c.diff()
            gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rsi = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
            out["rsi"] = _r(rsi.iloc[-1], 1)

        if len(c) > 35:   # MACD on % of price, so tiny-priced coins still show meaningful numbers
            macd = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c * 100
            sig = macd.ewm(span=9, adjust=False).mean()
            hst = macd - sig
            out["macd"] = _r(macd.iloc[-1], 3); out["macdSignal"] = _r(sig.iloc[-1], 3)
            out["macdHist"] = _r(hst.iloc[-1], 3); out["macdHistPrev"] = _r(hst.iloc[-2], 3)
            cross = None
            for k in range(1, 6):
                if hst.iloc[-k] > 0 >= hst.iloc[-k - 1]: cross = f"bull {k - 1}d ago"; break
                if hst.iloc[-k] < 0 <= hst.iloc[-k - 1]: cross = f"bear {k - 1}d ago"; break
            out["macdCross"] = cross

        if len(c) > 17:
            ll, hh = l.rolling(14).min(), h.rolling(14).max()
            k = 100 * (c - ll) / (hh - ll).replace(0, float("nan"))
            out["stochK"] = _r(k.iloc[-1], 1); out["stochD"] = _r(k.rolling(3).mean().iloc[-1], 1)

        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        if len(c) > 15:
            atr = tr.ewm(alpha=1/14, adjust=False).mean()
            out["atr"] = _rp(atr.iloc[-1]); out["atrPct"] = _r(atr.iloc[-1] / px * 100, 2)

        if len(c) > 20:
            m = c.rolling(20).mean(); sd = c.rolling(20).std()
            up, lo = m + 2 * sd, m - 2 * sd
            bw = (up - lo) / m * 100
            out["bbPctB"] = _r((px - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1]) * 100, 0)
            out["bbWidth"] = _r(bw.iloc[-1], 1)
            win = bw.dropna().tail(120)
            if len(win) >= 40:
                out["bbWidthPctile"] = _r((win < win.iloc[-1]).mean() * 100, 0)

        if len(c) > 30:
            upm, dnm = h.diff(), -l.diff()
            pdm = upm.where((upm > dnm) & (upm > 0), 0.0)
            ndm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
            atr_w = tr.ewm(alpha=1/14, adjust=False).mean()
            pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr_w
            ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr_w
            dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, float("nan"))
            out["adx"] = _r(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1], 1)
            out["plusDI"] = _r(pdi.iloc[-1], 1); out["minusDI"] = _r(ndi.iloc[-1], 1)

        mas = {}
        for key, ser in (("ema10", c.ewm(span=10, adjust=False).mean()), ("ema21", c.ewm(span=21, adjust=False).mean()),
                         ("sma20", c.rolling(20).mean()), ("sma50", c.rolling(50).mean()),
                         ("sma100", c.rolling(100).mean()), ("sma200", c.rolling(200).mean())):
            v = ser.iloc[-1]
            if v == v:
                mas[key] = {"v": _rp(v), "d": _r((px / v - 1) * 100, 1),
                            "slope": _r((v / ser.iloc[-6] - 1) * 100, 2) if len(ser) > 6 and ser.iloc[-6] == ser.iloc[-6] else None}
        out["mas"] = mas

        if len(c) > 20:
            ll, hh = l.rolling(14).min(), h.rolling(14).max()
            out["willR"] = _r(-100 * (hh.iloc[-1] - px) / (hh.iloc[-1] - ll.iloc[-1]), 1) if hh.iloc[-1] > ll.iloc[-1] else None
            tp = (h + l + c) / 3
            md = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
            out["cci"] = _r((tp.iloc[-1] - tp.rolling(20).mean().iloc[-1]) / (0.015 * md.iloc[-1]), 0) if md.iloc[-1] else None
            out["roc10"] = _r((px / c.iloc[-11] - 1) * 100, 1)
            out["roc20"] = _r((px / c.iloc[-21] - 1) * 100, 1)

        if len(c) > 20:
            v = hist["Volume"].astype(float)
            tp = (h + l + c) / 3
            rmf = tp * v
            pos = rmf.where(tp > tp.shift(), 0.0).rolling(14).sum()
            neg = rmf.where(tp < tp.shift(), 0.0).rolling(14).sum()
            out["mfi"] = _r(100 - 100 / (1 + pos.iloc[-1] / neg.iloc[-1]), 1) if neg.iloc[-1] else _r(100.0, 1)
            obv = (v * np.sign(c.diff().fillna(0))).cumsum()
            out["obvTrend"] = _r((obv.iloc[-1] - obv.iloc[-21]) / v.tail(20).mean(), 1) if v.tail(20).mean() else None

        if len(c) > 30:   # Parabolic SAR (0.02 / 0.2)
            af, mx = 0.02, 0.2
            bull = True; sar = l.iloc[0]; ep = h.iloc[0]; a = af; flip = 0
            for i in range(1, len(c)):
                sar = sar + a * (ep - sar)
                if bull:
                    sar = min(sar, l.iloc[i - 1], l.iloc[i - 2] if i > 1 else l.iloc[i - 1])
                    if l.iloc[i] < sar: bull, sar, ep, a, flip = False, ep, l.iloc[i], af, i
                    elif h.iloc[i] > ep: ep, a = h.iloc[i], min(a + af, mx)
                else:
                    sar = max(sar, h.iloc[i - 1], h.iloc[i - 2] if i > 1 else h.iloc[i - 1])
                    if h.iloc[i] > sar: bull, sar, ep, a, flip = True, ep, h.iloc[i], af, i
                    elif l.iloc[i] < ep: ep, a = l.iloc[i], min(a + af, mx)
            out["sar"] = _rp(sar); out["sarBull"] = bool(bull); out["sarAge"] = int(len(c) - 1 - flip)

        if len(c) > 52:
            tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
            kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
            spa = ((tenkan + kijun) / 2).shift(26)
            spb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
            top, bot = max(spa.iloc[-1], spb.iloc[-1]), min(spa.iloc[-1], spb.iloc[-1])
            if top == top and bot == bot:
                out["ichi"] = "above" if px > top else ("below" if px < bot else "inside")
                out["tkCross"] = "bull" if tenkan.iloc[-1] > kijun.iloc[-1] else "bear"

        N = 60
        def _ser(x, d=2): return [_r(v, d) for v in x.tail(N).tolist()]
        if len(c) > 15: out["rsiS"] = _ser(rsi, 1)
        if len(c) > 35: out["macdS"] = _ser(macd, 3); out["sigS"] = _ser(sig, 3); out["histS"] = _ser(hst, 3)
        if BENCH_CLOSE is not None:
            rl = (c / BENCH_CLOSE.reindex(c.index).ffill()).dropna()
            if len(rl) >= N:
                base = rl.iloc[-N]
                out["rsS"] = [_r(v / base * 100, 2) for v in rl.tail(N).tolist()]

        if len(c) > 30:   # support / resistance
            H, L, C = h.iloc[-1], l.iloc[-1], c.iloc[-1]
            P = (H + L + C) / 3
            out["pivots"] = {"P": _rp(P), "R1": _rp(2 * P - L), "R2": _rp(P + (H - L)), "R3": _rp(H + 2 * (P - L)),
                             "S1": _rp(2 * P - H), "S2": _rp(P - (H - L)), "S3": _rp(L - 2 * (H - P))}
            win_h, win_l = h.tail(60), l.tail(60)
            sh, sl = win_h.max(), win_l.min()
            upsw = win_h.idxmax() > win_l.idxmin()
            fib = {str(f): (_rp(sh - (sh - sl) * f) if upsw else _rp(sl + (sh - sl) * f)) for f in (0.236, 0.382, 0.5, 0.618, 0.786)}
            out["fib"] = {"hi": _rp(sh), "lo": _rp(sl), "up": bool(upsw), "lv": fib}
            hh_, ll_ = h.tail(120).tolist(), l.tail(120).tolist()
            highs = [hh_[i] for i in range(2, len(hh_) - 2) if hh_[i] == max(hh_[i - 2:i + 3])]
            lows = [ll_[i] for i in range(2, len(ll_) - 2) if ll_[i] == min(ll_[i - 2:i + 3])]
            lv = []
            for v in sorted(highs + lows):
                if lv and abs(v / lv[-1][0] - 1) < 0.015:
                    lv[-1] = ((lv[-1][0] * lv[-1][1] + v) / (lv[-1][1] + 1), lv[-1][1] + 1)
                else:
                    lv.append((v, 1))
            out["res"] = [{"v": _rp(v), "n": n} for v, n in sorted([x for x in lv if x[0] > px], key=lambda x: x[0])[:3]]
            out["sup"] = [{"v": _rp(v), "n": n} for v, n in sorted([x for x in lv if x[0] < px], key=lambda x: -x[0])[:3]]

        if len(c) > 60:
            v = hist["Volume"].astype(float)
            ex = {}
            n = 25
            ex["aroonUp"] = _r(h.tail(n + 1).values.argmax() / n * 100, 0)
            ex["aroonDn"] = _r(l.tail(n + 1).values.argmin() / n * 100, 0)
            bp = c - pd.concat([l, c.shift()], axis=1).min(axis=1)
            trr = pd.concat([h, c.shift()], axis=1).max(axis=1) - pd.concat([l, c.shift()], axis=1).min(axis=1)
            a7, a14, a28 = (bp.rolling(k).sum() / trr.rolling(k).sum() for k in (7, 14, 28))
            ex["uo"] = _r(100 * (4 * a7.iloc[-1] + 2 * a14.iloc[-1] + a28.iloc[-1]) / 7, 1)
            dd = c.diff().tail(14); su, sd_ = dd.clip(lower=0).sum(), -dd.clip(upper=0).sum()
            ex["cmo"] = _r(100 * (su - sd_) / (su + sd_), 0) if (su + sd_) else None
            vmp = (h - l.shift()).abs().rolling(14).sum(); vmm = (l - h.shift()).abs().rolling(14).sum()
            trs = trr.rolling(14).sum()
            ex["viPlus"] = _r(vmp.iloc[-1] / trs.iloc[-1], 2); ex["viMinus"] = _r(vmm.iloc[-1] / trs.iloc[-1], 2)
            e13 = c.ewm(span=13, adjust=False).mean()
            ex["bullPow"] = _r((h.iloc[-1] - e13.iloc[-1]) / px * 100, 2); ex["bearPow"] = _r((l.iloc[-1] - e13.iloc[-1]) / px * 100, 2)
            fi = (c.diff() * v).ewm(span=13, adjust=False).mean()
            dv = (c * v).tail(50).mean()
            ex["force"] = _r(fi.iloc[-1] / dv * 100, 2) if dv else None
            mid = (h + l) / 2
            ao = mid.rolling(5).mean() - mid.rolling(34).mean()
            ex["ao"] = _r(ao.iloc[-1] / px * 100, 2); ex["aoUp"] = bool(ao.iloc[-1] > ao.iloc[-2])
            mfm = ((c - l) - (h - c)) / (h - l).replace(0, float("nan"))
            mfv = (mfm * v).fillna(0)
            ex["cmf"] = _r(mfv.tail(20).sum() / v.tail(20).sum(), 3) if v.tail(20).sum() else None
            adl = mfv.cumsum()
            cho = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
            ex["chaikinOsc"] = _r(cho.iloc[-1] / v.tail(20).mean(), 2) if v.tail(20).mean() else None
            e3 = c.ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean()
            trix = e3.pct_change() * 100
            ex["trix"] = _r(trix.iloc[-1], 3); ex["trixSig"] = _r(trix.ewm(span=9, adjust=False).mean().iloc[-1], 3)
            y = np.log(c.tail(50).values); x = np.arange(len(y))
            sl_, ic_ = np.polyfit(x, y, 1); pred = sl_ * x + ic_
            ss = ((y - y.mean()) ** 2).sum()
            ex["lrSlope"] = _r((math.exp(min(sl_ * YEAR, 50)) - 1) * 100, 0)
            ex["lrR2"] = _r(1 - ((y - pred) ** 2).sum() / ss, 2) if ss else None
            tp_ = (h + l + c) / 3
            for k in (20, 50):
                vw = (tp_ * v).tail(k).sum() / v.tail(k).sum() if v.tail(k).sum() else None
                ex[f"vwap{k}"] = _rp(vw) if vw else None
            out["ex"] = ex

        ret = c.pct_change().dropna()
        if len(ret) > 60:
            ann = math.sqrt(YEAR)
            out["hv20"] = _r(ret.tail(20).std() * ann * 100, 1)
            out["hv60"] = _r(ret.tail(60).std() * ann * 100, 1)
            out["hv250"] = _r(ret.std() * ann * 100, 1)
            ddser = c / c.cummax() - 1
            out["maxDD"] = _r(ddser.min() * 100, 1); out["curDD"] = _r(ddser.iloc[-1] * 100, 1)
            out["adr20"] = _r(((h - l) / c).tail(20).mean() * 100, 2)
            r60 = ret.tail(60)
            out["upDays60"] = _r((r60 > 0).mean() * 100, 0)
            out["bestDay60"] = _r(r60.max() * 100, 1); out["worstDay60"] = _r(r60.min() * 100, 1)
            streak = 0
            for v in reversed(ret.tolist()):
                if v > 0 and streak >= 0: streak += 1
                elif v < 0 and streak <= 0: streak -= 1
                else: break
            out["streak"] = streak
            o_ = hist["Open"].astype(float)
            gaps = (o_ / c.shift() - 1).tail(60).dropna()
            out["gapUps"] = int((gaps > 0.02).sum()); out["gapDowns"] = int((gaps < -0.02).sum())
            if BENCH_CLOSE is not None:
                sret = BENCH_CLOSE.reindex(c.index).ffill().pct_change()
                j = pd.concat([ret, sret], axis=1).dropna()
                if len(j) > 60 and j.iloc[:, 1].var() > 0:
                    out["beta"] = _r(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var(), 2)
                    out["corr"] = _r(j.iloc[:, 0].corr(j.iloc[:, 1]), 2)
                    b1 = BENCH_CLOSE.tail(YEAR)
                    out["spyMaxDD"] = _r((b1 / b1.cummax() - 1).min() * 100, 1)

        if len(c) > 21:
            em = c.ewm(span=20, adjust=False).mean().iloc[-1]
            at10 = tr.ewm(alpha=1/10, adjust=False).mean().iloc[-1]
            out["keltPos"] = _r((px - (em - 2 * at10)) / (4 * at10) * 100, 0) if at10 else None
            dh, dl = h.iloc[-21:-1].max(), l.iloc[-21:-1].min()
            out["donHi"] = _rp(dh); out["donLo"] = _rp(dl)
            out["donBreak"] = "up" if px > dh else ("down" if px < dl else None)

        if BENCH_CLOSE is not None:   # coin return minus index return over 3, 6 and 12 months
            b = BENCH_CLOSE.reindex(c.index).ffill()
            for n, key in ((90, "rs3m"), (180, "rs6m"), (len(c) - 1, "rs12m")):
                if len(c) > n and n > 0 and b.iloc[-1 - n] == b.iloc[-1 - n]:
                    out[key] = _r(((c.iloc[-1] / c.iloc[-1 - n] - 1) - (b.iloc[-1] / b.iloc[-1 - n] - 1)) * 100, 1)
            ratio = (c / b).dropna()
            if len(ratio) >= 50:
                out["rsLineHigh"] = bool(ratio.iloc[-1] >= ratio.tail(YEAR).max() * 0.995)
    except Exception as e:
        out["error"] = str(e)[:80]
    return out


# ── Score one coin ────────────────────────────────────────────────────────────
def score_coin(key, hist):
    try:
        closes, volumes = hist["Close"], hist["Volume"]
        price, volume = float(closes.iloc[-1]), float(volumes.iloc[-1])
        if price <= 0:
            return None
        P = decision._p
        ma50 = P(closes.tail(50).mean())
        ma150 = P(closes.tail(150).mean() if len(closes) >= 150 else closes.mean())
        ma200 = P(closes.tail(200).mean() if len(closes) >= 200 else closes.mean())
        ma200p = P(closes.tail(230).head(200).mean()) if len(closes) >= 230 else ma200   # 200-day a month ago

        hi52, lo52 = float(hist["High"].max()), float(hist["Low"].min())
        pct_hi = round(max(0, (hi52 - price) / hi52 * 100), 1) if hi52 > 0 else 0
        pct_lo = round((price - lo52) / lo52 * 100, 1) if lo52 > 0 else 0

        def perf(n):
            if len(closes) > n:
                old = float(closes.iloc[-1 - n])
                return round((price - old) / old * 100, 2) if old > 0 else 0
            return 0
        chg1d = perf(1)
        chg5d, chg10d, chg20d, chg60d = perf(5), perf(10), perf(20), perf(60)
        chg250d = perf(min(YEAR, len(closes) - 1))           # 12-month change (field name shared with the share screeners)

        avg50v = float(volumes.tail(50).mean())
        volr = round(volume / avg50v, 2) if avg50v > 0 else 1.0

        c_ma50, c_ma150, c_ma200 = price > ma50, ma50 > ma150, ma150 > ma200
        c_trend, c_high, c_low, c_vol = ma200 > ma200p, pct_hi <= 25, pct_lo >= 25, volr >= VOL_BREAKOUT
        sepa = sum([c_ma50, c_ma150, c_ma200, c_trend, c_high, c_low, c_vol])
        new = len(closes) < 200          # new listing: cannot pass the 150/200-day checks yet, so always kept
        if sepa < SEPA_MIN and not new:
            return None

        vcp = min(sum([abs(chg5d) < abs(chg10d), abs(chg10d) < abs(chg20d), abs(chg20d) < abs(chg60d), volr < 0.8]), 4)
        ve = volr - 1.0
        pvr = min(round(abs(chg1d) / ve, 2) if ve > 0.1 else round(abs(chg1d) * 2, 2), 9.99)

        if sepa >= 5 and c_vol and vcp >= VCP_MIN: status = "breakout"
        elif sepa >= 4 and vcp >= VCP_MIN and pct_hi <= 15: status = "near-pivot"
        else: status = "watch"

        tail60 = hist.tail(60)
        ohlcv = [[P(r.Open), P(r.High), P(r.Low), P(r.Close), int(r.Volume)] for r in tail60.itertuples()]

        # Minervini stage (1-4), same rules as the share screeners
        up_vol = down_vol = 0.0
        for j in range(1, min(61, len(closes))):
            if float(closes.iloc[-j]) - float(closes.iloc[-j - 1]) > 0: up_vol += float(volumes.iloc[-j])
            else: down_vol += float(volumes.iloc[-j])
        acc_dist_ratio = round(up_vol / down_vol, 2) if down_vol > 0 else 99.0
        ret_s = closes.pct_change().dropna()
        v20 = float(ret_s.tail(20).std()) if len(ret_s) >= 20 else 0
        v40 = float(ret_s.tail(60).head(40).std()) if len(ret_s) >= 60 else v20
        vol_expanding = (v20 > v40 * 1.2) if v40 > 0 else False
        if len(closes) >= 60:
            stage4_pattern = (float(closes.tail(20).max()) < float(closes.tail(60).head(40).max()) * 0.97 and
                              float(closes.tail(20).min()) < float(closes.tail(60).head(40).min()) * 0.97)
        else:
            stage4_pattern = False
        pa200, ma15_a200, ma200_up = price > ma200, ma150 > ma200, ma200 > ma200p
        if pa200 and ma15_a200 and ma200_up and up_vol > down_vol: stage = 2
        elif not ma200_up and not pa200 and stage4_pattern: stage = 4
        elif vol_expanding or (pa200 and not ma200_up) or (not pa200 and not stage4_pattern and not ma15_a200): stage = 3
        else: stage = 1

        candle_signals = []
        if len(closes) >= 15:
            o, hgh, low = hist["Open"], hist["High"], hist["Low"]
            c_op, c_hi, c_lo, c_cl, c_vo = (float(x.iloc[-1]) for x in (o, hgh, low, closes, volumes))
            p_hi, p_lo, p_cl = float(hgh.iloc[-2]), float(low.iloc[-2]), float(closes.iloc[-2])
            avg_vol_10 = float(volumes.tail(10).mean())
            if c_cl > c_op and c_cl > p_cl:
                max_down = max([float(volumes.iloc[-k]) for k in range(2, 12)
                                if float(closes.iloc[-k]) < float(closes.iloc[-k - 1])] or [0])
                if max_down > 0 and c_vo > max_down and c_vo > avg_vol_10:
                    candle_signals.append("Pocket Pivot (Accumulation)")
            if c_hi < p_hi and c_lo > p_lo and c_vo < avg_vol_10 * 0.6:
                candle_signals.append("Inside Day & Vol Dry-up")
            avg_spread = float((hgh - low).tail(15).mean())
            if c_vo > avg_vol_10 * 1.5 and (c_hi - c_lo) < avg_spread * 0.8:
                candle_signals.append("Support Squat (Hidden Accumulation)" if c_cl >= c_op else "Resistance Squat (Hidden Distribution)")
            if c_lo < float(low.tail(6).head(5).min()) and c_cl > c_lo + (c_hi - c_lo) * 0.6 and c_vo > avg_vol_10:
                candle_signals.append("Shakeout & Reversal (Supply Absorbed)")

        return {
            "_key": key,
            "price": P(price), "change": round(chg1d, 2),
            "ma50": ma50, "ma150": ma150, "ma200": ma200,
            "volRatio": volr, "pvr": pvr, "vcpScore": vcp, "sepaScore": sepa,
            "pctFromHigh": pct_hi, "pctAboveLow": pct_lo,
            "hi52": P(hi52), "lo52": P(lo52),
            "chg5d": chg5d, "chg60d": chg60d, "chg250d": chg250d,
            "status": status, "stage": stage, "ohlcv": ohlcv,
            "candleSignals": candle_signals, "accDistRatio": acc_dist_ratio,
            "tech": technicals(hist),
            "ipo": new, "sessions": int(len(closes)),
            "listDate": str(hist.index[0].date()) if new else None,
            "checks": {"ma50": c_ma50, "ma150": c_ma150, "ma200": c_ma200,
                       "trend": c_trend, "high": c_high, "low": c_low, "vol": c_vol},
            # share-screener fields the dashboard expects; coins have no revenue, earnings or analysts
            "revGrowth": None, "revTrend": None, "revQuarters": [], "epsGrowth": None, "netMargin": None,
            "trailingEps": None, "forwardEps": None, "fundScore": 0,
            "nextEarnings": None, "nextEventLabel": None, "nextExDiv": None, "daysToEvent": None,
        }
    except Exception as e:
        print(f"  score {key}: {type(e).__name__}: {e}")
        return None


def describe(r, coin, src):
    """Coin metadata, token supply figures and the written summary."""
    mc = float(coin.get("market_cap") or 0)
    fdv = float(coin.get("fully_diluted_valuation") or 0)
    circ, tot, mx = coin.get("circulating_supply"), coin.get("total_supply"), coin.get("max_supply")
    cap_supply = mx or tot
    circ_pct = round(circ / cap_supply * 100, 1) if circ and cap_supply else None
    r.update({
        "ticker": coin["display"], "name": coin["name"], "cgId": coin["id"],
        "rank": coin.get("market_cap_rank"), "sector": coin["sector"],
        "mktcap": int(mc), "mktcapFmt": fmt_cap(mc), "fdv": int(fdv) if fdv else None,
        "mcFdv": round(mc / fdv * 100, 1) if fdv else None,
        "circPct": min(circ_pct, 100.0) if circ_pct is not None else None,
        "maxSupply": mx, "athPct": round(coin["ath_change_percentage"], 1) if coin.get("ath_change_percentage") is not None else None,
        "ath": decision._p(coin.get("ath")) if coin.get("ath") else None,
        "athDate": (coin.get("ath_date") or "")[:10] or None,
        "vol24": int(coin.get("total_volume") or 0),
        "volMcap": round(float(coin.get("total_volume") or 0) / mc * 100, 1) if mc else None,
        "chg7d": round(coin["price_change_percentage_7d_in_currency"], 1) if coin.get("price_change_percentage_7d_in_currency") is not None else None,
        "chg30d": round(coin["price_change_percentage_30d_in_currency"], 1) if coin.get("price_change_percentage_30d_in_currency") is not None else None,
        "exchange": src[0], "pair": src[1],
    })
    sigs = []
    if r["checks"]["vol"]: sigs.append(f"Vol {r['volRatio']}x avg")
    if r["chg250d"] > 50: sigs.append(f"+{r['chg250d']:.0f}% 12M")
    if r["vcpScore"] >= 3: sigs.append("VCP tightening")
    if r["pctFromHigh"] < 5: sigs.append("Near 52W high")
    if r["circPct"] is not None and r["circPct"] < 50: sigs.append(f"Only {r['circPct']:.0f}% of supply out")
    if not sigs: sigs.append(f"SEPA {r['sepaScore']}/7")
    r["shortSignal"] = " · ".join(sigs[:3])

    c = r["checks"]
    tr = "strongly uptrending" if r["chg250d"] > 40 else "uptrending" if r["chg250d"] > 15 else \
         "recovering" if r["chg250d"] > -15 else "in a downtrend"
    mas = "fully aligned (Price>MA50>MA150>MA200)" if (c["ma50"] and c["ma150"] and c["ma200"]) else "partially aligned"
    vls = f"Volume is {r['volRatio']}x the 50-day average" if r["volRatio"] >= 1.5 else f"Volume at {r['volRatio']}x average"
    vcpd = ["no base", "early base", "developing VCP", "good VCP (volume drying up)", "textbook VCP (volume at lows)"][r["vcpScore"]]
    supply = ""
    if r["circPct"] is not None:
        supply = (f" {r['circPct']:.0f}% of the maximum supply is circulating" +
                  (" - future unlocks can add selling pressure." if r["circPct"] < 70 else "."))
    ath = f" {abs(r['athPct']):.0f}% below its all-time high." if r["athPct"] is not None and r["athPct"] < -1 else \
          " At or near its all-time high." if r["athPct"] is not None else ""
    r["analysis"] = (f"{r['name']} is {tr} over 12 months ({fmt_pct(r['chg250d'])}), MAs {mas}. "
                     f"Forming {vcpd}, {r['pctFromHigh']}% below the 52W high of {fmt_price(r['hi52'])}. "
                     f"{vls}. SEPA {r['sepaScore']}/7, PVR {r['pvr']}.{ath}{supply}")
    return r


# ── Main ──────────────────────────────────────────────────────────────────────
META = {}


def fetch_all():
    global BENCH_CLOSE, META
    print("Loading CoinGecko market data...")
    markets = load_markets()
    cats = load_categories()

    coins, skipped, seen = [], {}, {}
    for c in markets:
        why = excluded_reason(c, cats)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        c["sector"] = sector_of(c["id"], cats)
        sym = c["symbol"].upper()
        seen[sym] = seen.get(sym, 0) + 1
        c["display"] = sym if seen[sym] == 1 else f"{sym}-{seen[sym]}"
        c["key"] = c["id"]
        coins.append(c)
    print(f"Eligible coins: {len(coins)} of {len(markets)} (excluded: {skipped})")

    print("Loading exchange market lists...")
    ex = Exchanges()
    # Fetch candles in rank order until UNIVERSE_SIZE coins have usable history
    hist_map, src_map, no_data, pegged = {}, {}, [], []
    candidates = coins[: int(UNIVERSE_SIZE * 1.6)]

    done = [0]
    t0 = time.time()

    def _one(c):
        got = ex.history(c["symbol"].upper(), float(c["current_price"]))
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  candles: {done[0]}/{len(candidates)} coins checked ({time.time() - t0:.0f}s)", flush=True)
        return c, got
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, candidates))
    universe = []
    for c, got in results:
        if len(universe) >= UNIVERSE_SIZE:
            break
        if not got:
            no_data.append(c["display"])
            continue
        df, exch, market = got
        r90 = df["Close"].pct_change().tail(90)
        if len(r90) >= 30 and r90.std() * math.sqrt(YEAR) * 100 < PEG_VOL_MAX:
            pegged.append(c["display"])
            continue
        hist_map[c["key"]], src_map[c["key"]] = df, (exch, market)
        universe.append(c)
    by_ex = {}
    for e, _ in src_map.values():
        by_ex[e] = by_ex.get(e, 0) + 1
    print(f"Universe: {len(universe)} coins with daily candles {by_ex}; "
          f"skipped {len(no_data)} with no supported market, {len(pegged)} pegged")
    if no_data:
        print("  no candles:", ", ".join(no_data[:40]))
    if len(universe) < UNIVERSE_SIZE * 0.5:
        raise RuntimeError(f"Only {len(universe)} coins have candle data - aborting rather than publishing a thin dashboard.")

    BENCH_CLOSE, bench_members = build_index(universe, hist_map)
    print(f"{BENCH_NAME}: {', '.join(bench_members)} ({len(BENCH_CLOSE) if BENCH_CLOSE is not None else 0} days)")

    rows = []
    for c in universe:
        hist = hist_map[c["key"]]
        r = score_coin(c["key"], hist.tail(YEAR))     # scoring uses the last year, as the share screeners do
        if r:
            rows.append(describe(r, c, src_map[c["key"]]))
    print(f"Passed SEPA filters: {len(rows)} of {len(universe)}")

    market = decision.market_state(BENCH_CLOSE)
    print(f"Market filter: {market['label']}")
    by_key = {c["key"]: c for c in universe}
    for r in rows:
        key = r.pop("_key")
        hist = hist_map[key]
        try:
            tail = hist.tail(20)
            adv = float((tail["Close"] * tail["Volume"]).mean()) if len(tail) else 0
            r["adv"] = int(adv)
            r["decision"] = decision.decide(hist, BENCH_CLOSE, market, liquid=adv >= MIN_ADV, locked=False, row=r)
            r["patterns"] = decision.candle_patterns(hist)
            r["ichimoku"] = decision.ichimoku(hist)
        except Exception as e:
            print(f"Decision failed for {r.get('ticker')}: {type(e).__name__}: {e}")
            r["decision"] = None
    bt = decision.backtest({k: hist_map[k] for k in by_key}, BENCH_CLOSE, min_adv=MIN_ADV)
    print(f"Backtest of the BUY rule: {bt}")
    META = {"market": market, "backtest": bt, "benchMembers": bench_members,
            "universe": len(universe), "noData": no_data, "pegged": pegged, "byExchange": by_ex,
            "excluded": sum(skipped.values())}
    from collections import Counter
    print("Decisions:", dict(Counter((r.get("decision") or {}).get("action") for r in rows)))
    rows.sort(key=lambda x: (x["sepaScore"], x["vcpScore"]), reverse=True)
    return rows


if __name__ == "__main__":
    print(f"SEPA+VCP Crypto Screener - {now_melb():%Y-%m-%d %H:%M} (Melbourne)")
    data = fetch_all()
    b = sum(1 for r in data if r["status"] == "breakout")
    p = sum(1 for r in data if r["status"] == "near-pivot")
    w = sum(1 for r in data if r["status"] == "watch")
    print(f"Results: {len(data)} | Breakouts:{b} | NearPivot:{p} | Watch:{w}")
    bad = sum(1 for r in data if not isinstance(r.get("price"), (int, float)) or not math.isfinite(r["price"]))
    if data and bad > len(data) * 0.05:
        print(f"ERROR: {bad}/{len(data)} rows have no valid price - refusing to publish a broken dashboard.")
        sys.exit(1)
    if not data:
        print("ERROR: screener returned 0 rows - refusing to overwrite the live dashboard with an empty one.")
        sys.exit(1)
    import build_dashboard
    html = build_dashboard.build(data, meta=META)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard built: {len(html):,} chars")
