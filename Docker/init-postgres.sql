-- PostgreSQL 初始化脚本
-- 由 docker-entrypoint-initdb.d 自动执行

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
