# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start dev server
npm run dev

# Type-check and build for production
npm run build

# Lint
npm run lint

# Preview production build
npm run preview
```

## Architecture

React 19 + TypeScript + Vite SPA. No React Router — navigation is pure state: `activePage` in `App.tsx` switches between `Overview`, `Portfolio`, `Watchlist`, and `Settings` page components.

**All market data is mocked.** There is no backend integration in the active `src/` code. `variable.ts` holds all static data (markets, holdings, transactions, news, alerts) and the seeded random generators that produce reproducible chart series. `App.tsx` runs two intervals: a 2.2 s ticker that applies random price drift to each market, and a 1 s clock.

### Key files

| File | Role |
|---|---|
| `src/variable.ts` | Single source of truth for all data and domain helpers (`buildMarket`, `genSeries`, `findMarket`, `NavItems`) |
| `src/types.ts` | All shared domain types (`Market`, `Holding`, `Transaction`, `Alert`, `SectorKey`, `RangeKey`, `Tone`, …) |
| `src/primitives.tsx` | Reusable SVG chart primitives: `Sparkline`, `LineChart` (with hover tooltip + volume bars via ResizeObserver), `Donut`, `Pill` |
| `src/format.ts` | Numeric formatters: `fmtPrice`, `fmtPct`, `fmtMoney`, `fmtCompact` |
| `src/App.tsx` | Root state, live ticking, trade modal state, alert list, watchlist Set, page routing |
| `src/TradeModal.tsx` | BUY/SELL order modal (MARKET/LIMIT/STOP); paper trading only |

### Styling

Custom CSS only (`src/App.css`) — no Tailwind. Dark green palette defined as CSS custom properties (`--bg0`–`--bg3`, `--green`, `--red`, `--yellow`, `--blue`, `--purple`, `--border`, etc.). Typography: Inter (sans) + IBM Plex Mono (mono, used for all numeric data).

Tonal styling flows through the `Tone` type (`"up" | "down" | "neutral" | "warn"`) and CSS class names like `.tone-up`, `.tone-down` applied on the `Pill` primitive and table rows.

### `src-secondary/`

Legacy/alternative implementation. Not built or imported by the active app — ignore it unless explicitly working on it.

## Constraints

- `LineChart` in `primitives.tsx` measures its container width via `ResizeObserver`; always render it inside a sized parent or it will collapse.
- Volume bars in `LineChart` are generated from a seeded algorithm (`seedRandom`) — they are not real volume data.
- The `Recharts` dependency in `package.json` exists but is not actively used; custom SVG primitives are used instead.
- `tsconfig.app.json` has `noUnusedLocals` and `noUnusedParameters` enabled — unused imports will break the build.
