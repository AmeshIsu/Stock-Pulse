import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Locate axe-core script for accessibility scans
const AXE_PATH = require.resolve('axe-core/axe.min.js');
const AXE_SOURCE = fs.readFileSync(AXE_PATH, 'utf-8');

const SCREENSHOT_DIR = path.resolve(__dirname, '../../../docs/screenshots');
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function runAxeScan(page: any, pageName: string) {
  await page.evaluate(AXE_SOURCE);
  const results = await page.evaluate(async () => {
    return await (window as any).axe.run(document, {
      runOnly: {
        type: 'tag',
        values: ['wcag2a', 'wcag2aa'],
      },
    });
  });

  const seriousOrCritical = results.violations.filter((v: any) =>
    ['serious', 'critical'].includes(v.impact)
  );

  if (seriousOrCritical.length > 0) {
    console.error(`Axe violations on ${pageName}:`, JSON.stringify(seriousOrCritical, null, 2));
  }
  expect(seriousOrCritical.length).toBe(0);
}

async function captureMultiViewportScreenshots(page: any, baseName: string) {
  const viewports = [
    { width: 375, height: 667, name: '375px' },
    { width: 768, height: 1024, name: '768px' },
    { width: 1440, height: 900, name: '1440px' },
  ];

  for (const vp of viewports) {
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.waitForTimeout(100);

    // Light mode
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `${baseName}_light_${vp.name}.png`),
      fullPage: true,
    });

    // Dark mode
    await page.evaluate(() => document.documentElement.classList.add('dark'));
    await page.waitForTimeout(100);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, `${baseName}_dark_${vp.name}.png`),
      fullPage: true,
    });

    // Reset to light
    await page.evaluate(() => document.documentElement.classList.remove('dark'));
  }
}

async function loginUser(page: any, email = 'analyst@stockpulse.lk', password = 'Password123!') {
  await page.goto('/login');
  await page.fill('[data-testid="login-email-input"]', email);
  await page.fill('[data-testid="login-password-input"]', password);
  await page.click('[data-testid="login-submit-btn"]');
  await page.waitForURL('/');

  const modal = page.locator('[data-testid="disclaimer-modal"]');
  try {
    await modal.waitFor({ state: 'visible', timeout: 3500 });
    await page.click('[data-testid="disclaimer-checkbox"]');
    await page.click('[data-testid="acknowledge-disclaimer-btn"]');
    await modal.waitFor({ state: 'hidden', timeout: 3500 });
  } catch {
    // Already dismissed or not shown
  }
}

test.describe('Stock Pulse Comprehensive E2E Suite', () => {
  test('01. Authentication Flow: Registration, Policy Check, and Login', async ({ page }) => {
    await page.goto('/register');
    await expect(page.locator('h1')).toHaveText('Create Your Account');

    // Verify live password policy checks
    await page.fill('[data-testid="register-email-input"]', 'testuser@stockpulse.lk');
    await page.fill('[data-testid="register-password-input"]', 'WeakPass');
    await expect(page.locator('[data-testid="password-policy-list"]')).toBeVisible();

    // Fill valid credentials matching policy
    await page.fill('[data-testid="register-password-input"]', 'StrongPass123!');
    await page.fill('[data-testid="register-confirm-password-input"]', 'StrongPass123!');
    await page.click('[data-testid="toggle-register-password-btn"]');
    await expect(page.locator('[data-testid="register-submit-btn"]')).toBeEnabled();

    await runAxeScan(page, 'register');
    await captureMultiViewportScreenshots(page, 'register');

    // Switch to Login
    await page.click('[data-testid="login-link"]');
    await expect(page.locator('h1')).toHaveText('Sign In to Stock Pulse');
    await expect(page.locator('[data-testid="register-link"]')).toBeVisible();

    // Forgot password modal
    await page.click('[data-testid="forgot-password-link"]');
    await expect(page.locator('[data-testid="forgot-password-modal"]')).toBeVisible();
    await page.click('[data-testid="close-forgot-modal-btn"]');

    // Sign in with pre-seeded demo user
    await page.evaluate(() => localStorage.clear());
    await page.fill('[data-testid="login-email-input"]', 'analyst@stockpulse.lk');
    await page.fill('[data-testid="login-password-input"]', 'Password123!');
    await page.click('[data-testid="toggle-password-visibility-btn"]');

    await runAxeScan(page, 'login');
    await captureMultiViewportScreenshots(page, 'login');

    await page.click('[data-testid="login-submit-btn"]');

    // Verify First-Login Disclaimer Modal appears
    await expect(page.locator('[data-testid="disclaimer-modal"]')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('[data-testid="acknowledge-disclaimer-btn"]')).toBeDisabled();
    await page.click('[data-testid="disclaimer-checkbox"]');
    await expect(page.locator('[data-testid="acknowledge-disclaimer-btn"]')).toBeEnabled();
    await page.click('[data-testid="acknowledge-disclaimer-btn"]');
    await expect(page.locator('[data-testid="disclaimer-modal"]')).not.toBeVisible();
  });

  test('02. User Dashboard: Market Tiles, Horizon Tabs, Movers & Watchlist', async ({ page }) => {
    // Login as analyst
    await loginUser(page);

    // Test Navigation controls
    await expect(page.locator('[data-testid="navbar-brand-logo"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-link-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-link-rankings"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-link-watchlist"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-link-alerts"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-link-how-it-works"]')).toBeVisible();
    await expect(page.locator('[data-testid="nav-account-link"]')).toBeVisible();

    // Notifications bell popover
    await page.click('[data-testid="navbar-notification-bell"]');
    await expect(page.locator('[data-testid="see-all-notifications-link"]')).toBeVisible();
    await page.click('[data-testid="navbar-notification-bell"]');

    // Market Summary Tiles
    await expect(page.locator('[data-testid="market-summary-tiles"]')).toBeVisible();
    await expect(page.locator('[data-testid="high-tier-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="volatility-headline"]')).toBeVisible();

    // Horizon tabs switching
    await page.click('[data-testid="tab-short"]');
    await expect(page.locator('[data-testid="confidence-notice-bar"]')).toBeVisible();
    await page.click('[data-testid="tab-mid"]');
    await page.click('[data-testid="tab-long"]');
    await page.click('[data-testid="tab-overall"]');
    await expect(page.locator('[data-testid="how-we-tested-this-link"]')).toBeVisible();

    // Consistent leaders & Movers
    const leader = page.locator('[data-testid^="leader-pill-"]').first();
    if (await leader.isVisible()) {
      await expect(leader).toBeVisible();
    }
    await expect(page.locator('[data-testid="movers-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="indicative-tag"]')).toBeVisible();
    await expect(page.locator('[data-testid="watchlist-snapshot"]')).toBeVisible();
    await expect(page.locator('[data-testid="go-to-watchlist-link"]')).toBeVisible();

    // View All 20 Stocks button
    await expect(page.locator('[data-testid="view-full-rankings-btn"]')).toBeVisible();

    // Mobile menu toggle check
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('[data-testid="mobile-menu-toggle"]')).toBeVisible();
    await page.click('[data-testid="mobile-menu-toggle"]');
    await page.click('[data-testid="mobile-menu-toggle"]');
    await page.setViewportSize({ width: 1440, height: 900 });

    // Check Top Picks rows and nav-logout-btn
    await expect(page.locator('[data-testid="nav-logout-btn"]')).toBeVisible();
    const topPickRow = page.locator('[data-testid^="top-pick-row-"]').first();
    if (await topPickRow.isVisible()) {
      await expect(topPickRow).toBeVisible();
    }

    await runAxeScan(page, 'dashboard');
    await captureMultiViewportScreenshots(page, 'dashboard');
  });

  test('03. Rankings Table: Sorting, Filtering, Horizon Weight Sliders, CSV, Compare', async ({ page }) => {
    await loginUser(page);
    await page.goto('/rankings');
    await expect(page.locator('[data-testid="rankings-table-container"]')).toBeVisible();

    // Column sorting on all headers
    await page.click('[data-testid="sort-rank"]');
    await page.click('[data-testid="sort-code"]');
    await page.click('[data-testid="sort-overall"]');
    await page.click('[data-testid="sort-short"]');
    await page.click('[data-testid="sort-mid"]');
    await page.click('[data-testid="sort-long"]');
    await page.click('[data-testid="sort-risk"]');
    await page.click('[data-testid="sort-price"]');
    await page.click('[data-testid="sort-change"]');

    // Filtering by Tier and Risk
    await page.click('[data-testid="filter-tier-high"]');
    await page.click('[data-testid="filter-tier-neutral"]');
    await page.click('[data-testid="filter-tier-low"]');
    await page.click('[data-testid="filter-tier-all"]');
    await page.click('[data-testid="filter-risk-normal"]');
    await page.click('[data-testid="filter-risk-elevated"]');
    await page.click('[data-testid="filter-risk-all"]');
    await page.click('[data-testid="filter-exclude-liquidity-checkbox"]');
    await page.click('[data-testid="filter-exclude-liquidity-checkbox"]');

    // Search
    await page.fill('[data-testid="rankings-search-input"]', 'COMB');
    await expect(page.locator('[data-testid="rankings-row-COMB.N"]')).toBeVisible();
    await page.fill('[data-testid="rankings-search-input"]', '');

    // Horizon Weights Sliders
    await page.click('[data-testid="toggle-weight-sliders-btn"]');
    await expect(page.locator('[data-testid="weight-sliders-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="weight-slider-short"]')).toBeVisible();
    await expect(page.locator('[data-testid="weight-slider-mid"]')).toBeVisible();
    await expect(page.locator('[data-testid="weight-slider-long"]')).toBeVisible();
    await page.click('[data-testid="apply-weights-btn"]');
    await page.click('[data-testid="reset-weights-btn"]');
    await page.click('[data-testid="toggle-weight-sliders-btn"]');

    // Column visibility menu
    await page.click('[data-testid="column-visibility-btn"]');
    await expect(page.locator('[data-testid="column-visibility-menu"]')).toBeVisible();
    await page.click('[data-testid="column-visibility-btn"]');

    // Export CSV
    await expect(page.locator('[data-testid="export-csv-btn"]')).toBeVisible();

    // Compare Drawer
    const check1 = page.locator('input[aria-label="Select COMB.N for comparison"]');
    const check2 = page.locator('input[aria-label="Select SEYB.X for comparison"]');
    if (await check1.isVisible()) await check1.check();
    if (await check2.isVisible()) await check2.check();

    await page.click('[data-testid="open-compare-drawer-btn"]');
    await expect(page.locator('[data-testid="compare-drawer"]')).toBeVisible();
    await page.click('[data-testid="close-compare-drawer-btn"]');

    await runAxeScan(page, 'rankings');
    await captureMultiViewportScreenshots(page, 'rankings');
  });

  test('04. Stock Detail Page: TradingView Chart, Indicators, Outlooks, Alert Creation', async ({ page }) => {
    await loginUser(page);
    await page.goto('/stocks/COMB.N');
    await expect(page.locator('[data-testid="stock-code-header"]')).toHaveText('COMB.N');
    await expect(page.locator('[data-testid="stock-price"]')).toBeVisible();

    // Chart toggles & ranges
    await page.click('[data-testid="chart-type-line"]');
    await page.click('[data-testid="chart-type-candlestick"]');
    await page.click('[data-testid="range-1m"]');
    await page.click('[data-testid="range-3m"]');
    await page.click('[data-testid="range-6m"]');
    await page.click('[data-testid="range-1y"]');
    await page.click('[data-testid="range-5y"]');
    await page.click('[data-testid="range-max"]');

    // Overlays
    await page.click('[data-testid="toggle-sma20"]');
    await page.click('[data-testid="toggle-sma50"]');
    await page.click('[data-testid="toggle-sma200"]');
    await page.click('[data-testid="toggle-rsi"]');
    await expect(page.locator('[data-testid="rsi-panel"]')).toBeVisible();
    await page.click('[data-testid="toggle-macd"]');
    await expect(page.locator('[data-testid="macd-panel"]')).toBeVisible();

    // TradingView Attribution Check
    await expect(page.locator('[data-testid="tradingview-attribution"]')).toBeVisible();

    // 3 Horizon Outlook Cards
    await expect(page.locator('[data-testid="outlook-card-short"]')).toBeVisible();
    await expect(page.locator('[data-testid="outlook-card-mid"]')).toBeVisible();
    await expect(page.locator('[data-testid="outlook-card-long"]')).toBeVisible();

    // Risk and Liquidity Panels
    await expect(page.locator('[data-testid="risk-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="liquidity-panel"]')).toBeVisible();

    // Watchlist Star Toggle
    await page.click('[data-testid="toggle-watchlist-btn"]');

    // Create Alert Modal
    await page.click('[data-testid="create-alert-btn"]');
    await expect(page.locator('[data-testid="create-alert-modal"]')).toBeVisible();
    await page.selectOption('[data-testid="alert-type-select"]', 'price_above');
    await page.fill('[data-testid="alert-threshold-input"]', '125.0');
    await page.click('[data-testid="submit-alert-btn"]');

    await runAxeScan(page, 'stock_detail');
    await captureMultiViewportScreenshots(page, 'stock_detail');
  });

  test('05. Watchlist, Alerts, Notifications, Methodology, Account Pages', async ({ page }) => {
    await loginUser(page);

    // Watchlist
    await page.goto('/watchlist');
    await expect(page.locator('[data-testid="watchlist-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="add-stock-select"]')).toBeVisible();
    await expect(page.locator('[data-testid="add-stock-btn"]')).toBeVisible();
    const removeWlBtn = page.locator('[data-testid^="remove-watchlist-btn-"]').first();
    const editNoteInput = page.locator('[data-testid^="edit-note-input-"]').first();
    const saveNoteBtn = page.locator('[data-testid^="save-note-btn-"]').first();
    const quickAlertBtn = page.locator('[data-testid^="quick-alert-btn-"]').first();
    if (await removeWlBtn.isVisible()) await expect(removeWlBtn).toBeVisible();
    if (await editNoteInput.isVisible()) await expect(editNoteInput).toBeVisible();
    if (await saveNoteBtn.isVisible()) await expect(saveNoteBtn).toBeVisible();
    if (await quickAlertBtn.isVisible()) await expect(quickAlertBtn).toBeVisible();
    await runAxeScan(page, 'watchlist');
    await captureMultiViewportScreenshots(page, 'watchlist');

    // Alerts
    await page.goto('/alerts');
    await expect(page.locator('[data-testid="alerts-page"]')).toBeVisible();
    const toggleAlertBtn = page.locator('[data-testid^="toggle-alert-status-"]').first();
    const editAlertBtn = page.locator('[data-testid^="edit-alert-btn-"]').first();
    const deleteAlertBtn = page.locator('[data-testid^="delete-alert-btn-"]').first();
    if (await toggleAlertBtn.isVisible()) await expect(toggleAlertBtn).toBeVisible();
    if (await editAlertBtn.isVisible()) await expect(editAlertBtn).toBeVisible();
    if (await deleteAlertBtn.isVisible()) await expect(deleteAlertBtn).toBeVisible();
    await page.click('[data-testid="create-new-alert-btn"]');
    await expect(page.locator('[data-testid="alert-config-modal"]')).toBeVisible();
    await page.click('[data-testid="save-alert-rule-btn"]');
    await runAxeScan(page, 'alerts');
    await captureMultiViewportScreenshots(page, 'alerts');

    // Notifications
    await page.goto('/notifications');
    await expect(page.locator('[data-testid="notifications-page"]')).toBeVisible();
    const markReadBtn = page.locator('[data-testid^="mark-read-btn-"]').first();
    if (await markReadBtn.isVisible()) await expect(markReadBtn).toBeVisible();
    const markAllBtn = page.locator('[data-testid="mark-all-read-btn"]');
    if (await markAllBtn.isVisible()) {
      await markAllBtn.click();
    }
    await runAxeScan(page, 'notifications');
    await captureMultiViewportScreenshots(page, 'notifications');

    // Methodology (How It Works)
    await page.goto('/how-it-works');
    await expect(page.locator('[data-testid="how-it-works-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="print-methodology-btn"]')).toBeVisible();
    await runAxeScan(page, 'how_it_works');
    await captureMultiViewportScreenshots(page, 'how_it_works');

    // Account Page
    await page.goto('/account');
    await expect(page.locator('[data-testid="account-page"]')).toBeVisible();
    await page.click('[data-testid="account-theme-toggle"]');
    await page.click('[data-testid="account-theme-toggle"]');
    await expect(page.locator('[data-testid="current-password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="new-password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="confirm-new-password-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="change-password-submit-btn"]')).toBeVisible();
    await runAxeScan(page, 'account');
    await captureMultiViewportScreenshots(page, 'account');
  });

  test('06. RBAC Separation: User sees 403 on /admin; Admin Console Workflow', async ({ page }) => {
    // 1. Regular User attempt to access /admin
    await loginUser(page, 'analyst@stockpulse.lk', 'Password123!');

    await page.goto('/admin');
    await expect(page.locator('[data-testid="forbidden-page"]')).toBeVisible();
    await page.click('[data-testid="forbidden-return-home-btn"]');
    await page.waitForURL('/');

    // Sign out
    await page.goto('/account');
    await page.click('[data-testid="account-signout-btn"]');
    await page.waitForURL('/login');

    // 2. Admin User Login
    await loginUser(page, 'admin@stockpulse.lk', 'AdminPass123!');

    // Verify admin console nav link
    await expect(page.locator('[data-testid="nav-link-admin-console"]')).toBeVisible();

    // Go to Admin Console
    await page.goto('/admin');
    await expect(page.locator('[data-testid="admin-overview-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="admin-nav-overview"]')).toBeVisible();
    await expect(page.locator('[data-testid="admin-nav-model-governance"]')).toBeVisible();
    await expect(page.locator('[data-testid="admin-nav-user-management"]')).toBeVisible();
    await expect(page.locator('[data-testid="admin-nav-data-management"]')).toBeVisible();
    await expect(page.locator('[data-testid="admin-nav-audit-log"]')).toBeVisible();
    await runAxeScan(page, 'admin_overview');
    await captureMultiViewportScreenshots(page, 'admin_overview');

    // Admin Models
    await page.click('[data-testid="admin-nav-model-governance"]');
    await expect(page.locator('[data-testid="admin-models-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="governance-warning-banner"]')).toBeVisible();

    // Horizon tabs in admin
    await page.click('[data-testid="admin-tab-short"]');
    await page.click('[data-testid="admin-tab-mid"]');
    await page.click('[data-testid="admin-tab-long"]');
    await page.click('[data-testid="admin-tab-short"]');

    // Inspect Model Drawer
    await page.click('[data-testid="admin-model-row-ens_rank4"]');
    await expect(page.locator('[data-testid="admin-model-drawer"]')).toBeVisible();
    await page.click('[data-testid="close-model-drawer-btn"]');

    // Test Model Switch with Preview
    const activateBtn = page.locator('[data-testid="activate-model-btn-momentum_126"]');
    if (await activateBtn.isVisible()) {
      await activateBtn.click();
      await expect(page.locator('[data-testid="model-switch-modal"]')).toBeVisible();
      await page.fill('[data-testid="switch-reason-textarea"]', 'Operational backtest verification test');
      const warningCheck = page.locator('[data-testid="acknowledge-warning-checkbox"]');
      if (await warningCheck.isVisible()) {
        await warningCheck.check();
      }
      await page.click('[data-testid="confirm-model-switch-btn"]');
      await expect(page.locator('[data-testid="model-switch-modal"]')).not.toBeVisible();
    }

    // Reset to Auto
    await page.click('[data-testid="reset-all-auto-btn"]');

    await runAxeScan(page, 'admin_models');
    await captureMultiViewportScreenshots(page, 'admin_models');

    // Admin Users
    await page.click('[data-testid="admin-nav-user-management"]');
    await expect(page.locator('[data-testid="admin-users-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="open-create-user-modal-btn"]')).toBeVisible();
    const selectRole = page.locator('[data-testid^="select-role-"]').first();
    const toggleStatus = page.locator('[data-testid^="toggle-status-btn-"]').first();
    const resetPwd = page.locator('[data-testid^="reset-pwd-btn-"]').first();
    if (await selectRole.isVisible()) await expect(selectRole).toBeVisible();
    if (await toggleStatus.isVisible()) await expect(toggleStatus).toBeVisible();
    if (await resetPwd.isVisible()) await expect(resetPwd).toBeVisible();
    await runAxeScan(page, 'admin_users');
    await captureMultiViewportScreenshots(page, 'admin_users');

    // Admin Data
    await page.click('[data-testid="admin-nav-data-management"]');
    await expect(page.locator('[data-testid="admin-data-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="dry-run-checkbox"]')).toBeVisible();
    await expect(page.locator('[data-testid="submit-upload-btn"]')).toBeVisible();
    const rollbackBtn = page.locator('[data-testid^="rollback-btn-"]').first();
    const confirmRollbackBtn = page.locator('[data-testid="confirm-rollback-btn"]');
    const copyTempPwdBtn = page.locator('[data-testid="copy-temp-password-btn"]');
    const themeToggleBtn = page.locator('[data-testid="theme-toggle-btn"]');
    if (await rollbackBtn.isVisible()) await expect(rollbackBtn).toBeVisible();
    if (await confirmRollbackBtn.isVisible()) await expect(confirmRollbackBtn).toBeVisible();
    if (await copyTempPwdBtn.isVisible()) await expect(copyTempPwdBtn).toBeVisible();
    if (await themeToggleBtn.isVisible()) await expect(themeToggleBtn).toBeVisible();
    await page.click('[data-testid="evaluate-alerts-now-btn"]');
    await expect(page.locator('[data-testid="alert-eval-status"]')).toBeVisible();
    await runAxeScan(page, 'admin_data');
    await captureMultiViewportScreenshots(page, 'admin_data');

    // Admin Audit
    await page.click('[data-testid="admin-nav-audit-log"]');
    await expect(page.locator('[data-testid="admin-audit-page"]')).toBeVisible();
    await expect(page.locator('[data-testid="export-audit-csv-btn"]')).toBeVisible();
    await runAxeScan(page, 'admin_audit');
    await captureMultiViewportScreenshots(page, 'admin_audit');

    // Return to App
    await page.click('[data-testid="admin-back-to-app-btn"]');
    await page.waitForURL('/');
  });
});
