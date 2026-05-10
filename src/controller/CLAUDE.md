# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the server
go run . startserver

# Connect to database only (for testing DB connectivity)
go run . connectdb

# Build binary
go build -o server .

# Build Docker image
docker build -t green-pulse-controller .

# Run Docker container (requires myEnv.env mounted)
docker run -p 8080:8080 green-pulse-controller

# Python stock fetcher (called internally by Go, but can be run manually)
cd script && python3 fetch.py <SYMBOL> <DB_URL> <DB_PASSWORD> <REGION>
```

## Environment Setup

Copy `myEnv.env.example` to `myEnv.env` and fill in:
- `supaDBpass` — Supabase database password
- `database` — Supabase project ID (used in connection string)
- `ClientID` — Google OAuth client ID
- `ClientSecret` — Google OAuth client secret

The app loads from `myEnv.env` (not `.env`) via `godotenv`.

## Architecture

This is a Go HTTP server on port 8080 that serves as the backend for the green-pulse stock tracking app.

### Request Flow

1. **Auth** — Google OAuth 2.0 (Authorization Code Grant). `cfunc.go` handles `/signin`, `/signup`, `/callback`. On callback, a token + user info are passed as query params to the frontend redirect URL (`http://127.0.0.1:5500/src/controller/frontendtest/index.html`).
2. **Stock Data** — `/watchlist?symbol=SYMBOL` checks the DB first; if missing, spawns `python3 script/fetch.py` to pull data from Yahoo Finance via yfinance and insert into PostgreSQL. Returns JSON with `symbol`, `count`, and `records[]`.
3. **Favorites** — `/favorites?user_id=ID&symbol=SYMBOL` supports GET/POST/DELETE for per-user symbol bookmarks.

### Stock Refresh Scheduler

`scheduleStockRefreshes()` in `main.go` ticks every 5 minutes and, if it's a weekday after 4:05 PM NYC time, calls `RefreshAllStocks()` which re-fetches every tracked symbol via the Python script.

### Database

PostgreSQL on Supabase (ap-southeast-1). Connection is pooled via `pgxpool` (max 10 connections). Simple query protocol is enabled — required for Supabase's pgBouncer pooler (pgx prepared statements are incompatible with it).

Inferred tables: `users` (google_id, email), `stocks` (symbol + OHLCV data), `favorites` (user_id + symbol).

### Package Layout

| Package | File | Responsibility |
|---|---|---|
| `main` | `main.go` | Route registration, scheduler, CLI entrypoint |
| `function` | `cfunc.go` | Google OAuth config and callback handling |
| `function` | `dbfunc.go` | DB pool, user CRUD, stock queries, Python subprocess |
| `function` | `wfunc.go` | HTTP handlers for `/watchlist` and `/favorites` |
| *(Python)* | `script/fetch.py` | yfinance → PostgreSQL ingestion |

### Frontend Test UI

`frontendtest/` contains a standalone HTML+JS app (no build step) for testing all endpoints. Run it with VS Code Live Server or any static file server on port 5500. Auth tokens are stored in `localStorage`.

## Key Constraints

- The Python fetcher exits with code `2` for unknown symbols — `wfunc.go` maps this to a 404 "unknown symbol" response.
- All endpoints return `Access-Control-Allow-Origin: *` (CORS open).
- The OAuth redirect URL is hardcoded to `localhost:8080` and the frontend to `127.0.0.1:5500` — changing either requires editing `cfunc.go` and `test.js`.
- `go.mod` specifies `go 1.26.1` but the Dockerfile builds with `golang:1.23-alpine`.
