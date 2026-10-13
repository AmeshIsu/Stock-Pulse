# Stock Pulse — Frontend Web Application

The frontend client for **Stock Pulse**, providing an institutional-grade upside ranking and market analytics dashboard for Colombo Stock Exchange (CSE / S&P SL20) equities.

---

## 1. Technology Stack

- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS with custom design tokens, dark mode support (`class`), and Inter tabular typography
- **Navigation**: React Router DOM v7
- **State & Caching**: TanStack Query (React Query)
- **Forms & Validation**: `react-hook-form` + `zod`
- **Charting**:
  - TradingView Lightweight Charts (with library attribution) for interactive price charts
  - Recharts for distribution, factor gauge, and administrative diagnostic charts
- **Markdown**: `react-markdown` + `remark-gfm`
- **Testing**: Vitest + Testing Library + Playwright + `axe-core`

---

## 2. Directory Structure

```
app/frontend/
├── src/
│   ├── components/         # Shared UI components, layout, navigation, modals
│   │   ├── common/         # Gauges, badges, sparklines, confidence pills, theme toggle
│   │   ├── layout/         # AppLayout, Navbar, Footer, FreshnessBanner, DisclaimerModal
│   │   ├── charts/         # TradingView price chart, indicator panels
│   │   └── ui/             # Dialogs, drawers, buttons, dropdowns, tabs
│   ├── contexts/           # AuthContext, ThemeContext
│   ├── lib/                # API client with CSRF & auto-refresh, utils
│   ├── pages/              # User-facing routes
│   │   ├── Dashboard.tsx
│   │   ├── Rankings.tsx
│   │   ├── StockDetail.tsx
│   │   ├── Watchlist.tsx
│   │   ├── Alerts.tsx
│   │   ├── Notifications.tsx
│   │   ├── HowItWorks.tsx
│   │   ├── Account.tsx
│   │   ├── Login.tsx
│   │   ├── Register.tsx
│   │   ├── Forbidden.tsx
│   │   └── NotFound.tsx
│   ├── pages/admin/        # Code-split admin console routes (prefixed admin-*)
│   │   ├── AdminLayout.tsx
│   │   ├── AdminOverview.tsx
│   │   ├── AdminModels.tsx
│   │   ├── AdminUsers.tsx
│   │   ├── AdminData.tsx
│   │   ├── AdminAudit.tsx
│   │   └── AdminMethodology.tsx
│   ├── App.tsx             # Route definitions with lazy loading & guards
│   ├── main.tsx            # Application entry point
│   └── index.css           # Tailwind base, utilities, font features
├── dist/                   # Production build outputs
├── tests/                  # Vitest unit tests & Playwright E2E suites
├── vite.config.ts          # Vite configuration with dev proxy & chunk splitting
├── tailwind.config.js      # Design tokens, color scales, tier definitions
└── package.json
```

---

## 3. Development & Build Commands

All commands are run from `app/frontend/`:

```bash
# Install dependencies
npm install

# Start Vite dev server (proxies /api to http://127.0.0.1:8000)
npm run dev

# Run unit tests
npm test

# Build production bundle
npm run build

# Verify that no model names or metrics leaked into user bundles
node ../../scripts/check_bundle_leaks.mjs
```

---

## 4. User Area vs. Admin Area Boundaries

To preserve strict compliance with methodology governance:
1. **Forbidden Terms in User UI**: User views, tooltips, and public methodology texts never expose internal model identifiers (`momentum_126`, `ridge`, `rf_`, `lgbm`, `ens_rank4`, `lstm`, `tabpfn`) or statistical terms (`IC`, `null band`, `SHAP`, `coefficient`, `model_id`, `Holm`).
2. **Code Splitting**: All administrative components are dynamically imported and bundled into isolated chunks (`admin-*`).
3. **Automated Verification**: The post-build script `scripts/check_bundle_leaks.mjs` scans all non-admin chunks and fails CI if any model terms are discovered.
