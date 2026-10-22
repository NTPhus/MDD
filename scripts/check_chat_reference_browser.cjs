'use strict';
// Compatibility entry point. Changed words regenerate the same bot turn;
// unchanged words confirm/reassess only. Former unconditional checks are obsolete.
// The replacement suite mocks all jobs/voices and never runs model inference,
// training, paid APIs or physical microphone/speaker checks.
const { spawnSync } = require('node:child_process');
const path = require('node:path');

function run() {
  console.log('Reference UI checks now run check_reference_regeneration_browser.cjs.');
  console.log('Report: data/generated/chat-ui-cleanup-qa/browser-summary.json');
  const result = spawnSync(process.execPath, [path.join(__dirname, 'check_reference_regeneration_browser.cjs')], {
    cwd: path.resolve(__dirname, '..'), env: process.env, stdio: 'inherit',
  });
  if (result.error) throw result.error;
  process.exitCode = result.status === null ? 1 : result.status;
}

module.exports = { run };
if (require.main === module) {
  try { run(); } catch (error) { console.error(error); process.exitCode = 1; }
}
