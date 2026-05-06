-- ==============================================================================
-- v007_create_admin_users.sql
-- ==============================================================================
-- Manual migration for company admin login accounts.
-- Safe to re-run. It also brings older admin_users tables from db/init/10 into
-- the current MVP shape where possible.
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

ALTER TABLE admin_users
    ALTER COLUMN password_hash TYPE VARCHAR(255);

ALTER TABLE admin_users
    ADD COLUMN IF NOT EXISTS name VARCHAR(100);

UPDATE admin_users
SET name = LEFT(email, 100)
WHERE name IS NULL;

ALTER TABLE admin_users
    ALTER COLUMN name SET NOT NULL;

ALTER TABLE admin_users
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

UPDATE admin_users
SET is_active = TRUE
WHERE is_active IS NULL;

ALTER TABLE admin_users
    ALTER COLUMN is_active SET DEFAULT TRUE;

UPDATE admin_users
SET role = 'staff'
WHERE role IS NULL OR role NOT IN ('owner', 'admin', 'staff', 'manager', 'agent');

ALTER TABLE admin_users
    ALTER COLUMN role SET DEFAULT 'owner',
    ALTER COLUMN role SET NOT NULL;

ALTER TABLE admin_users
    DROP CONSTRAINT IF EXISTS admin_users_role_check;

ALTER TABLE admin_users
    ADD CONSTRAINT admin_users_role_check
    CHECK (role IN ('owner', 'admin', 'staff', 'manager', 'agent'));

CREATE INDEX IF NOT EXISTS idx_admin_users_tenant_id ON admin_users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_admin_users_email ON admin_users(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_users_email_lower ON admin_users(LOWER(email));
