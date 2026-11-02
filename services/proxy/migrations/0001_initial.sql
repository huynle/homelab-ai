-- 0001: users, API keys, daily usage, admin sessions, audit log.
-- Forward-only: never edit a released migration; add the next number instead.
-- Timestamps are UTC ISO 8601 text, "YYYY-MM-DDTHH:MM:SSZ" (sorts lexically). Days are "YYYY-MM-DD".

CREATE TABLE users (
    email           TEXT PRIMARY KEY,                       -- lowercased
    role            TEXT NOT NULL CHECK (role IN ('owner', 'member')),
    invited_by      TEXT,
    invited_at      TEXT,
    removed_at      TEXT,
    last_sign_in_at TEXT
);

CREATE TABLE api_keys (
    public_id      TEXT PRIMARY KEY CHECK (length(public_id) = 12),  -- the key id; dk_<public_id>_<secret>
    key_hmac       TEXT NOT NULL UNIQUE,                    -- hex HMAC-SHA256(pepper, full key); never the key
    name           TEXT NOT NULL,
    owner_email    TEXT NOT NULL REFERENCES users (email),
    source         TEXT NOT NULL CHECK (source IN ('issued', 'imported')),
    prefix         TEXT NOT NULL,                           -- dk_<public_id>_… or imported…
    created_at     TEXT NOT NULL,
    expires_at     TEXT,
    revoked_at     TEXT,
    last_used_at   TEXT,
    rate_limit_rpm INTEGER                                  -- reserved; always NULL today
);
CREATE INDEX api_keys_owner ON api_keys (owner_email);

-- model is '' when no model was resolved (NULL would break the primary key's uniqueness for upserts);
-- the service layer maps '' back to null.
CREATE TABLE usage_daily (
    day            TEXT NOT NULL,
    public_id      TEXT NOT NULL REFERENCES api_keys (public_id),
    route          TEXT NOT NULL,
    model          TEXT NOT NULL DEFAULT '',
    requests       INTEGER NOT NULL DEFAULT 0,
    client_errors  INTEGER NOT NULL DEFAULT 0,
    server_errors  INTEGER NOT NULL DEFAULT 0,
    latency_ms_sum REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, public_id, route, model)
) WITHOUT ROWID;
CREATE INDEX usage_daily_key_day ON usage_daily (public_id, day);

CREATE TABLE sessions (
    id_hash         TEXT PRIMARY KEY,                       -- sha256 of the cookie value
    email           TEXT NOT NULL REFERENCES users (email),
    csrf_token_hash TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    revoked_at      TEXT
);
CREATE INDEX sessions_email ON sessions (email);

-- No IP addresses, tokens or keys: only emails, key ids and a small JSON detail object.
CREATE TABLE audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    at           TEXT NOT NULL,
    actor_email  TEXT,
    event        TEXT NOT NULL,
    target_email TEXT,
    key_id       TEXT,
    detail       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX audit_at ON audit (at);
