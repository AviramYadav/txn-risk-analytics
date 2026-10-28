-- Behavioural features per transaction, computed only from information
-- available at transaction time (windows look backwards, never forwards).

DROP TABLE IF EXISTS features;

CREATE TABLE features AS
WITH base AS (
    SELECT f.txn_id,
           f.customer_id,
           f.ts,
           CAST(strftime('%s', f.ts) AS INTEGER) AS ts_epoch,
           f.amount,
           f.channel,
           f.device_id,
           f.is_fraud,
           m.category,
           m.risk_tier,
           m.city      AS merchant_city,
           c.home_city
    FROM fct_transactions f
    JOIN dim_merchant m ON m.merchant_id = f.merchant_id
    JOIN dim_customer c ON c.customer_id = f.customer_id
),
windowed AS (
    SELECT base.*,
           COUNT(*)    OVER (PARTITION BY customer_id ORDER BY ts_epoch
                             RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW)       AS txn_count_1h,
           COUNT(*)    OVER (PARTITION BY customer_id ORDER BY ts_epoch
                             RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW)      AS txn_count_24h,
           SUM(amount) OVER (PARTITION BY customer_id ORDER BY ts_epoch
                             RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW)      AS amount_sum_24h,
           AVG(amount) OVER (PARTITION BY customer_id ORDER BY ts_epoch
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)   AS hist_avg_amount,
           COUNT(*)    OVER (PARTITION BY customer_id ORDER BY ts_epoch
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)   AS hist_txn_count,
           ts_epoch - LAG(ts_epoch) OVER (PARTITION BY customer_id ORDER BY ts_epoch) AS secs_since_prev,
           ROW_NUMBER() OVER (PARTITION BY customer_id, device_id ORDER BY ts_epoch)  AS device_use_seq
    FROM base
)
SELECT txn_id,
       customer_id,
       ts,
       is_fraud,
       amount,
       channel,
       category,
       risk_tier,
       CAST(strftime('%H', ts) AS INTEGER)                                     AS hour,
       CASE WHEN CAST(strftime('%H', ts) AS INTEGER) < 6 THEN 1 ELSE 0 END     AS is_night,
       CASE WHEN merchant_city <> 'ONLINE' AND merchant_city <> home_city
            THEN 1 ELSE 0 END                                                  AS is_out_of_city,
       txn_count_1h,
       txn_count_24h,
       amount_sum_24h,
       COALESCE(amount / NULLIF(hist_avg_amount, 0), 1.0)                      AS amount_to_avg_ratio,
       hist_txn_count,
       COALESCE(secs_since_prev, 30 * 86400)                                   AS secs_since_prev,
       CASE WHEN device_id IS NOT NULL AND device_use_seq = 1 THEN 1 ELSE 0 END AS is_new_device
FROM windowed;
