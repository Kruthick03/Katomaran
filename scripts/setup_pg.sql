DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'sentineluser') THEN
      CREATE ROLE sentineluser WITH LOGIN PASSWORD 'sentinelpass' SUPERUSER CREATEDB;
   ELSE
      ALTER ROLE sentineluser WITH PASSWORD 'sentinelpass' SUPERUSER CREATEDB;
   END IF;
END
$$;

SELECT 'CREATE DATABASE sentineltrack OWNER sentineluser'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'sentineltrack') \gexec

GRANT ALL PRIVILEGES ON DATABASE sentineltrack TO sentineluser;
