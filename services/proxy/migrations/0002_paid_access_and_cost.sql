-- 0002: per-key access to paid (OpenRouter) models, and upstream cost in daily usage.
-- allow_paid: 1 lets the key use registry models with adapter openrouter (POST /api/alpha/decisions);
-- only the owner changes it (PATCH /admin/api/keys/{id}). cost_usd_sum: OpenRouter's usage.cost, summed.

ALTER TABLE api_keys ADD COLUMN allow_paid INTEGER NOT NULL DEFAULT 0 CHECK (allow_paid IN (0, 1));
ALTER TABLE usage_daily ADD COLUMN cost_usd_sum REAL NOT NULL DEFAULT 0;
