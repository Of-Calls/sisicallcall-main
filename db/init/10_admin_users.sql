-- ==============================================================================
-- 10_admin_users.sql - company admin login accounts
-- ==============================================================================
-- Admin users are separate from customer face authentication sessions.
-- Seeded accounts are for local development only.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS admin_users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      UUID NOT NULL REFERENCES tenants(id),
    email          VARCHAR(255) UNIQUE NOT NULL,
    password_hash  VARCHAR(255) NOT NULL,
    name           VARCHAR(100) NOT NULL,
    role           VARCHAR(20) NOT NULL DEFAULT 'owner'
                   CHECK (role IN ('owner', 'admin', 'staff', 'manager', 'agent')),
    is_active      BOOLEAN DEFAULT TRUE,
    last_login_at  TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_admin_users_tenant_id ON admin_users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_admin_users_email ON admin_users(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_users_email_lower ON admin_users(LOWER(email));
