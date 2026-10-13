#!/usr/bin/env node
/**
 * scripts/check_bundle_leaks.mjs
 * Post-build verification script:
 * Inspects all non-admin client bundles to guarantee ZERO leakage of internal
 * model names, model metrics, or algorithmic governance details to regular users.
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, '..');
const DIST_ASSETS = path.join(PROJECT_ROOT, 'app', 'frontend', 'dist', 'assets');

const FORBIDDEN_WORDS = [
  'momentum_126',
  'ridge',
  'rf_',
  'lgbm',
  'ens_rank4',
  'LSTM',
  'TabPFN',
  'Holm',
  'null band',
  'model_id',
  'hit rate',
  'rank correlation',
];

if (!fs.existsSync(DIST_ASSETS)) {
  console.error(`[LEAK CHECK ERROR] Dist assets directory does not exist: ${DIST_ASSETS}`);
  console.error('Please run "npm run build" in app/frontend first.');
  process.exit(1);
}

const files = fs.readdirSync(DIST_ASSETS);
const jsFiles = files.filter((f) => f.endsWith('.js'));

console.log('='.repeat(65));
console.log('STOCK PULSE BUNDLE LEAK VERIFICATION');
console.log('='.repeat(65));
console.log(`Inspecting ${jsFiles.length} JavaScript asset chunks in: ${DIST_ASSETS}\n`);

let leakCount = 0;
let inspectedNonAdminChunks = 0;

for (const file of jsFiles) {
  // Admin-only code-split chunks are exempted
  if (file.startsWith('admin-')) {
    console.log(`  [EXEMPT ADMIN CHUNK] ${file}`);
    continue;
  }

  inspectedNonAdminChunks++;
  const filePath = path.join(DIST_ASSETS, file);
  const content = fs.readFileSync(filePath, 'utf-8');

  const fileLeaks = [];
  for (const word of FORBIDDEN_WORDS) {
    let regex;
    if (word === 'rf_') {
      // Must not match csrf_ (Cross-Site Request Forgery)
      regex = /(?<!cs)rf_[a-zA-Z0-9_]*/i;
    } else {
      regex = new RegExp(`\\b${word}\\b`, 'i');
    }

    if (regex.test(content)) {
      fileLeaks.push(word);
    }
  }

  if (fileLeaks.length > 0) {
    console.error(`  [FAIL - LEAK DETECTED] in public chunk: ${file}`);
    for (const leak of fileLeaks) {
      console.error(`    -> Forbidden model term found: "${leak}"`);
    }
    leakCount += fileLeaks.length;
  } else {
    console.log(`  [PASS - CLEAN] ${file}`);
  }
}

console.log('\n' + '-'.repeat(65));
if (leakCount > 0) {
  console.error(`RESULT: FAILED with ${leakCount} forbidden model/metric leaks in user bundle!`);
  process.exit(1);
} else {
  console.log(`RESULT: PASSED. All ${inspectedNonAdminChunks} user chunks are 100% clean.`);
  console.log('Zero internal model names or metric words leaked to user bundles.');
  console.log('='.repeat(65));
  process.exit(0);
}
