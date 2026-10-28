-- Business MI queries. Each block starts with "-- name: <query_name>".

-- name: fraud_by_channel
SELECT channel,
       COUNT(*)                                             AS txns,
       SUM(is_fraud)                                        AS fraud_txns,
       ROUND(100.0 * AVG(is_fraud), 3)                      AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount END))   AS fraud_value_inr
FROM fct_transactions
GROUP BY channel
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_category
SELECT m.category,
       m.risk_tier,
       COUNT(*)                                             AS txns,
       SUM(f.is_fraud)                                      AS fraud_txns,
       ROUND(100.0 * AVG(f.is_fraud), 3)                    AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN f.is_fraud = 1 THEN f.amount END)) AS fraud_value_inr
FROM fct_transactions f
JOIN dim_merchant m ON m.merchant_id = f.merchant_id
GROUP BY m.category, m.risk_tier
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_hour
SELECT CAST(strftime('%H', ts) AS INTEGER)                  AS hour,
       COUNT(*)                                             AS txns,
       SUM(is_fraud)                                        AS fraud_txns,
       ROUND(100.0 * AVG(is_fraud), 3)                      AS fraud_rate_pct
FROM fct_transactions
GROUP BY hour
ORDER BY hour;

-- name: fraud_by_type
SELECT fraud_type,
       COUNT(*)                                             AS fraud_txns,
       COUNT(DISTINCT customer_id)                          AS customers_hit,
       ROUND(AVG(amount))                                   AS avg_amount_inr,
       ROUND(SUM(amount))                                   AS total_value_inr
FROM fct_transactions
WHERE is_fraud = 1
GROUP BY fraud_type
ORDER BY total_value_inr DESC;

-- name: weekly_trend
SELECT strftime('%Y-W%W', ts)                               AS week,
       COUNT(*)                                             AS txns,
       ROUND(SUM(amount))                                   AS spend_inr,
       SUM(is_fraud)                                        AS fraud_txns,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)) AS fraud_value_inr
FROM fct_transactions
GROUP BY week
ORDER BY week;

-- name: segment_summary
SELECT c.segment,
       COUNT(DISTINCT c.customer_id)                        AS customers,
       COUNT(*)                                             AS txns,
       ROUND(AVG(f.amount))                                 AS avg_ticket_inr,
       ROUND(100.0 * AVG(f.is_fraud), 3)                    AS fraud_rate_pct
FROM fct_transactions f
JOIN dim_customer c ON c.customer_id = f.customer_id
GROUP BY c.segment
ORDER BY avg_ticket_inr DESC;
