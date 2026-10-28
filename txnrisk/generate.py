"""Synthetic card-transaction generator.

Produces customers, merchants and transactions with realistic everyday spending,
three labelled fraud scenarios, legitimate "noise" that looks risky (travel,
new phones, big purchases) and a small number of deliberately injected
data-quality defects so the downstream checks have something real to find.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai",
          "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Ahmedabad"]
CITY_WEIGHTS = np.array([0.17, 0.17, 0.14, 0.10, 0.10, 0.10, 0.08, 0.05, 0.05, 0.04])

# category -> (log-mean amount in INR, log-sigma, merchant risk tier, share of merchants that are online)
CATEGORIES = {
    "grocery":       (6.0, 0.6, "low",    0.10),
    "fuel":          (7.2, 0.4, "low",    0.00),
    "restaurants":   (6.5, 0.7, "low",    0.15),
    "utilities":     (7.4, 0.5, "low",    0.80),
    "apparel":       (7.5, 0.8, "medium", 0.40),
    "travel":        (8.6, 0.9, "medium", 0.70),
    "atm_cash":      (8.0, 0.5, "medium", 0.00),
    "electronics":   (9.0, 0.9, "high",   0.50),
    "gift_cards":    (7.8, 0.7, "high",   0.90),
    "digital_goods": (6.2, 1.0, "high",   0.95),
}
CATEGORY_WEIGHTS = {"grocery": 0.22, "fuel": 0.12, "restaurants": 0.18, "utilities": 0.08,
                    "apparel": 0.10, "travel": 0.04, "atm_cash": 0.08, "electronics": 0.05,
                    "gift_cards": 0.03, "digital_goods": 0.10}

# segment -> (share of customers, transaction-rate multiplier, amount multiplier)
SEGMENTS = {"mass": (0.60, 0.8, 1.0), "affluent": (0.30, 1.0, 1.4), "premium": (0.10, 1.3, 2.2)}
CREDIT_LIMITS = {"mass": (50_000, 150_000), "affluent": (150_000, 500_000), "premium": (500_000, 1_500_000)}

HOUR_WEIGHTS = np.array([1.0, 0.6, 0.4, 0.3, 0.3, 0.5, 1.5, 3.0, 4.5, 5.0, 5.5, 6.0,
                         7.0, 6.5, 6.0, 6.0, 6.5, 7.0, 8.0, 8.5, 8.0, 6.0, 4.0, 2.0])
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

FRAUD_SCENARIOS = ["account_takeover", "card_testing", "skimming"]
DEFECT_TYPES = ["null_amount", "negative_amount", "future_timestamp",
                "orphan_merchant", "invalid_channel", "duplicate_row"]


@dataclass
class GeneratorConfig:
    n_customers: int = 5000
    n_merchants: int = 800
    days: int = 90
    start_date: str = "2025-01-01"
    txns_per_day: float = 0.65
    online_share: float = 0.35
    fraud_customer_rate: float = 0.03
    travel_customer_rate: float = 0.10
    spree_customer_rate: float = 0.06
    unreported_fraud_rate: float = 0.08
    defect_rate: float = 0.002
    seed: int = 42


def generate(cfg: GeneratorConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    merchants = _merchants(cfg, rng)
    customers = _customers(cfg, rng)
    pools = _merchant_pools(merchants)
    legit = _legit_transactions(cfg, rng, customers, merchants, pools)
    fraud = _fraud_transactions(cfg, rng, customers, merchants, pools)

    txns = pd.concat([legit, fraud], ignore_index=True)
    txns = txns.sort_values("ts", kind="mergesort").reset_index(drop=True)
    txns.insert(0, "txn_id", [f"T{i:08d}" for i in range(1, len(txns) + 1)])
    txns, defects = _inject_defects(cfg, rng, txns)
    txns["ts"] = txns["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"customers": customers, "merchants": merchants, "transactions": txns, "defects": defects}


def _merchants(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    cats = list(CATEGORIES)
    weights = np.array([CATEGORY_WEIGHTS[c] for c in cats])
    category = rng.choice(cats, size=cfg.n_merchants, p=weights / weights.sum())
    online_share = np.array([CATEGORIES[c][3] for c in category])
    online = rng.random(cfg.n_merchants) < online_share
    city = np.where(online, "ONLINE", rng.choice(CITIES, size=cfg.n_merchants, p=CITY_WEIGHTS))
    return pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(1, cfg.n_merchants + 1)],
        "category": category,
        "risk_tier": [CATEGORIES[c][2] for c in category],
        "city": city,
    })


def _customers(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEGMENTS)
    segment = rng.choice(names, size=cfg.n_customers, p=[SEGMENTS[s][0] for s in names])
    start = pd.Timestamp(cfg.start_date)
    signup = start - pd.to_timedelta(rng.integers(30, 2000, cfg.n_customers), unit="D")
    limits = [rng.integers(*CREDIT_LIMITS[s]) // 1000 * 1000 for s in segment]
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, cfg.n_customers + 1)],
        "segment": segment,
        "home_city": rng.choice(CITIES, size=cfg.n_customers, p=CITY_WEIGHTS),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "credit_limit": np.array(limits, dtype=float),
    })


def _merchant_pools(merchants: pd.DataFrame) -> dict:
    city, cat = merchants["city"].to_numpy(), merchants["category"].to_numpy()
    online = city == "ONLINE"
    idx = np.arange(len(merchants))
    return {
        "all": idx,
        "online": idx[online],
        "physical": {c: idx[city == c] for c in CITIES},
        "atm": {c: idx[(city == c) & (cat == "atm_cash")] for c in CITIES},
        "online_high_risk": idx[online & np.isin(cat, ["electronics", "gift_cards", "digital_goods"])],
        "online_digital": idx[online & (cat == "digital_goods")],
        "online_electronics": idx[online & (cat == "electronics")],
    }


def _amounts(rng, mu, sigma, mult=1.0):
    return np.round(np.exp(rng.normal(mu, sigma)) * mult, 2)


def _channel(m_city: np.ndarray, m_cat: np.ndarray) -> np.ndarray:
    return np.where(m_city == "ONLINE", "ECOM", np.where(m_cat == "atm_cash", "ATM", "POS"))


def _round_atm(amount: np.ndarray, channel: np.ndarray) -> np.ndarray:
    return np.where(channel == "ATM", np.maximum(100.0, np.round(amount / 100) * 100), amount)


def _legit_transactions(cfg, rng, customers, merchants, pools) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start_date)
    m_ids = merchants["merchant_id"].to_numpy()
    m_cat = merchants["category"].to_numpy()
    m_city = merchants["city"].to_numpy()
    mu = np.array([CATEGORIES[c][0] for c in m_cat])
    sigma = np.array([CATEGORIES[c][1] for c in m_cat])
    travellers = rng.random(len(customers)) < cfg.travel_customer_rate
    spreers = rng.random(len(customers)) < cfg.spree_customer_rate
    night_owls = rng.random(len(customers)) < 0.08
    night_p = np.roll(HOUR_P, -6)  # same shape, shifted six hours later

    parts = []
    for i, cust in enumerate(customers.itertuples(index=False)):
        _, rate_mult, amt_mult = SEGMENTS[cust.segment]
        n = rng.poisson(cfg.txns_per_day * cfg.days * rate_mult)
        if n == 0:
            continue
        day = rng.integers(0, cfg.days, n)
        hour = rng.choice(24, n, p=night_p if night_owls[i] else HOUR_P)
        offset = day * 86400 + hour * 3600 + rng.integers(0, 3600, n)

        online = rng.random(n) < cfg.online_share
        home = pools["physical"][cust.home_city]
        midx = np.where(online, rng.choice(pools["online"], n), rng.choice(home, n))
        elsewhere = ~online & (rng.random(n) < 0.03)
        midx[elsewhere] = rng.choice(pools["all"], elsewhere.sum())
        if travellers[i]:
            # A genuine trip: several days of physical spend in another city (looks like skimming, isn't).
            trip_city = rng.choice([c for c in CITIES if c != cust.home_city])
            d0, dur = rng.integers(0, cfg.days - 5), rng.integers(2, 6)
            on_trip = ~online & (day >= d0) & (day < d0 + dur)
            midx[on_trip] = rng.choice(pools["physical"][trip_city], on_trip.sum())

        if spreers[i]:
            # A genuine online shopping spree: a quick burst of high-risk purchases (looks like ATO, isn't).
            k = int(rng.integers(3, 7))
            t0 = int(rng.integers(0, cfg.days * 86400 - 3600))
            day = np.concatenate([day, np.full(k, t0 // 86400)])
            offset = np.concatenate([offset, t0 + np.sort(rng.integers(0, 3600, k))])
            midx = np.concatenate([midx, rng.choice(pools["online_high_risk"], k)])
            n += k

        channel = _channel(m_city[midx], m_cat[midx])
        amount = _round_atm(_amounts(rng, mu[midx], sigma[midx], amt_mult), channel)

        devices = [f"D{i:06d}A"] + ([f"D{i:06d}B"] if rng.random() < 0.3 else [])
        device = np.where(channel == "ECOM", rng.choice(devices, n), None).astype(object)
        new_phone = (channel == "ECOM") & (rng.random(n) < 0.03)
        device[new_phone] = [f"D{i:06d}N{k}" for k in range(new_phone.sum())]

        parts.append(pd.DataFrame({
            "customer_id": cust.customer_id,
            "merchant_id": m_ids[midx],
            "ts": start + pd.to_timedelta(offset, unit="s"),
            "amount": amount,
            "channel": channel,
            "device_id": device,
            "is_fraud": 0,
            "fraud_type": None,
        }))
    return pd.concat(parts, ignore_index=True)


def _fraud_transactions(cfg, rng, customers, merchants, pools) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start_date)
    m_ids = merchants["merchant_id"].to_numpy()
    m_cat = merchants["category"].to_numpy()
    m_city = merchants["city"].to_numpy()
    n_victims = max(1, round(cfg.fraud_customer_rate * len(customers)))
    victims = rng.choice(len(customers), size=n_victims, replace=False)

    parts = []
    for v in victims:
        cust = customers.iloc[v]
        scenario = rng.choice(FRAUD_SCENARIOS, p=[0.5, 0.3, 0.2])
        t0 = int(rng.integers(14 * 86400, (cfg.days - 1) * 86400))
        if rng.random() < 0.5:  # fraudsters favour the small hours
            t0 = t0 // 86400 * 86400 + int(rng.integers(0, 5 * 3600))

        if scenario == "account_takeover":
            k = int(rng.integers(3, 9))
            offset = t0 + np.sort(rng.integers(0, 2 * 3600, k))
            midx = rng.choice(pools["online_high_risk"], k)
            amount = _amounts(rng, np.full(k, 8.2), 0.8)
            # Malware / session hijack runs on the victim's own phone, so "new device" is no giveaway.
            device_id = f"D{v:06d}A" if rng.random() < 0.35 else f"X{v:06d}"
            device = np.full(k, device_id, dtype=object)
        elif scenario == "card_testing":
            small, large = int(rng.integers(4, 11)), int(rng.integers(1, 3))
            offset = np.concatenate([t0 + np.sort(rng.integers(0, 20 * 60, small)),
                                     t0 + 20 * 60 + np.sort(rng.integers(0, 30 * 60, large))])
            midx = np.concatenate([rng.choice(pools["online_digital"], small),
                                   rng.choice(pools["online_electronics"], large)])
            amount = np.concatenate([np.round(rng.uniform(1, 60, small), 2),
                                     _amounts(rng, np.full(large, 9.2), 0.5)])
            device = np.full(small + large, f"X{v:06d}", dtype=object)
        else:  # skimming: cloned card used in person, in a different city
            k = int(rng.integers(2, 6))
            city = rng.choice([c for c in CITIES if c != cust["home_city"]])
            offset = t0 + np.sort(rng.integers(0, 18 * 3600, k))
            atm = pools["atm"][city] if len(pools["atm"][city]) else pools["physical"][city]
            midx = np.where(rng.random(k) < 0.5, rng.choice(atm, k), rng.choice(pools["physical"][city], k))
            amount = _amounts(rng, np.full(k, 8.6), 0.5)
            device = np.full(k, None, dtype=object)

        channel = _channel(m_city[midx], m_cat[midx])
        # Some victims never dispute the charges, so the fraud never gets a label.
        reported = rng.random() >= cfg.unreported_fraud_rate
        parts.append(pd.DataFrame({
            "customer_id": cust["customer_id"],
            "merchant_id": m_ids[midx],
            "ts": start + pd.to_timedelta(offset, unit="s"),
            "amount": _round_atm(amount, channel),
            "channel": channel,
            "device_id": device,
            "is_fraud": int(reported),
            "fraud_type": scenario if reported else None,
        }))
    return pd.concat(parts, ignore_index=True)


def _inject_defects(cfg, rng, txns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrupt a small sample of legitimate rows the way real feeds break."""
    per_type = max(1, int(len(txns) * cfg.defect_rate / len(DEFECT_TYPES)))
    legit_idx = txns.index[txns["is_fraud"] == 0].to_numpy()
    chosen = rng.choice(legit_idx, size=per_type * 5, replace=False).reshape(5, per_type)

    txns = txns.copy()
    txns.loc[chosen[0], "amount"] = np.nan
    txns.loc[chosen[1], "amount"] = -txns.loc[chosen[1], "amount"]
    txns.loc[chosen[2], "ts"] = txns.loc[chosen[2], "ts"] + pd.Timedelta(days=400)
    txns.loc[chosen[3], "merchant_id"] = "M99999"
    txns.loc[chosen[4], "channel"] = "UNK"
    dupes = txns.loc[rng.choice(legit_idx, size=per_type, replace=False)]
    txns = pd.concat([txns, dupes], ignore_index=True)

    defects = pd.DataFrame({"defect_type": DEFECT_TYPES, "injected_rows": per_type})
    return txns, defects
