# Drop-in for your Vite project

Copy everything in this folder into your `src/` directory.

## File layout

```
src/
├── App.tsx              ← replaces your current App.tsx
├── App.css              ← replaces your current App.css
├── Sidebar.tsx          ← replaces component.tsx (delete component.tsx)
├── TradeModal.tsx
├── format.ts
├── primitives.tsx
├── types.ts
├── variable.ts          ← replaces your current variable.ts
└── pages/
    ├── Overview.tsx
    ├── Portfolio.tsx
    ├── Watchlist.tsx
    └── Settings.tsx
```

## Notes

1. **Delete `component.tsx`** — it's been renamed/replaced by `Sidebar.tsx`.
2. **Recharts is no longer needed.** All charts are pure SVG (`primitives.tsx`). You can `npm uninstall recharts`.
3. **Logo asset** — `App.tsx` no longer imports `./assets/greenPulseLogo.svg`. The sidebar now uses a CSS-drawn pulsing mark. If you want to keep your SVG logo, edit `Sidebar.tsx` and put `<img src={...} />` back inside `.logo`.
4. **Fonts** — Inter + IBM Plex Mono are loaded via `index.html` (you already have these). Confirm the `<link>` is still in your `index.html`.
5. **Globals removed** — your old `index.css` had a bunch of unrelated tokens (`--accent: #aa3bff`, light-mode vars, etc.). The new `App.css` declares its own `:root` so it doesn't collide. You can leave `index.css` as-is or strip it down.
6. **TypeScript strict** — types live in `types.ts`. If `tsc` complains, check that your `tsconfig.json` has `"jsx": "react-jsx"` and `"moduleResolution": "bundler"` (Vite default).

## What's in here
- 4 pages: Overview, Portfolio, Watchlist, Settings
- Live-ticking prices (every 2.2s)
- Click-any-ticker → loads in main chart
- Working time-range buttons (1D/1W/1M/3M/1Y/5Y) with crosshair tooltips
- Sortable + filterable watchlist with star-toggle
- Buy/Sell/Limit/Stop trade modal with confirmation
- Allocation donut, sparklines, news rail, alerts table
