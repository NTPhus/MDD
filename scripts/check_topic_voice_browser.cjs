'use strict';
// Real local config/provider samples; topic replies and paid TTS requests are fixtures.
const { chromium } = require(process.env.MDD_PLAYWRIGHT_MODULE || '../.tools/ui-qa/node_modules/playwright');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/topic-voice-qa');
(async () => {
 fs.mkdirSync(output, { recursive:true });
 const browser = await chromium.launch({headless:true, executablePath:process.env.MDD_CHROME_PATH || undefined, args:['--no-sandbox']});
 const page = await browser.newPage({viewport:{width:1440,height:1000}, reducedMotion:'reduce'});
 const errors=[], topicCalls=[], voiceCalls=[], checks=[];
 page.on('pageerror', e=>errors.push(e.message));
 try {
  const configResponse = await page.request.get(base + '/api/chat/config');
  assert.equal(configResponse.status(), 200);
  const config = await configResponse.json();
  assert.ok(config.english_voices.length >= 8);
  assert.ok(config.english_voices.some(v=>v.locale==='en-GB'));
  const readyConfig = {...config, speech_ready:true, models:config.models.map(model=>({...model,ready:true}))};
  await page.route('**/api/chat/config',route=>route.fulfill({json:readyConfig}));
  const sample = config.english_voices.find(v=>v.locale==='en-GB' && v.preview_url);
  const sampleResponse=await page.request.get(base+sample.preview_url);
  assert.equal(sampleResponse.status(),200);
  const sampleAudio=await sampleResponse.body();
  const jobId='fedcba9876543210fedcba9876543210';
  const reply='Welcome! What would you like to talk about today?';
  await page.route('**/api/chat/topic', route=>{
   topicCalls.push(route.request().postDataJSON());
   return route.fulfill({json:{id:jobId,status:'ready',text:reply,metadata:{input:'topic'}}});
  });
  await page.route('**/api/chat/voice', route=>{
   const payload=route.request().postDataJSON(); voiceCalls.push(payload);
   return route.fulfill({json:{id:jobId,status:'ready',text:reply,metadata:{},voice:{status:'ready',url:'/api/chat/audio/'+jobId,...payload}}});
  });
  await page.route('**/api/chat/audio/'+jobId, route=>route.fulfill({body:sampleAudio,contentType:'audio/mpeg'}));
  // Block all remaining writes so QA cannot submit inference or paid tasks.
  await page.route('**/api/chat/speech', route=>route.fulfill({status:500,json:{message:'QA blocks acoustic inference'}}));
  await page.goto(base+'/#chat');
  await page.waitForFunction(()=>window.MDDChat?.state.config?.english_voices?.length);
  const topic=page.locator('.chat-topic-card').first();
  await topic.click();
  await page.locator('#chat-topic-setup').waitFor({state:'visible'});
  assert.equal(topicCalls.length,0); assert.equal(voiceCalls.length,0);
  assert.ok((await page.locator('#topic-setup-role').textContent()).length>0);
  await page.screenshot({path:path.join(output,'voice-picker-desktop.png')});
  checks.push('Selecting a topic opens a briefing/voice dialog before any conversation or TTS request.');
  await page.locator('[data-voice-locale="en-GB"]').click();
  assert.ok(await page.locator('.topic-voice-row').count());
  assert.equal(await page.locator('.topic-voice-accent').filter({hasText:'US'}).count(),0);
  await page.locator('#topic-voice-search').fill(sample.name.split(' - ')[0]);
  await page.locator('.topic-voice-preview').first().click();
  await page.waitForFunction(()=>window.MDDChat.state.previewAudio?.duration>0);
  assert.equal(voiceCalls.length,0);
  const previewPlaying=await page.evaluate(()=>!window.MDDChat.state.previewAudio.paused);
  assert.equal(previewPlaying,true);
  await page.locator('input[name=topic-voice]').first().check();
  await page.locator('input[name=topic-speed][value="0.85"]').check();
  assert.equal(await page.evaluate(()=>window.MDDChat.state.previewAudio.playbackRate),.85);
  await page.locator('#topic-setup-start').click();
  await page.waitForFunction(()=>document.body.classList.contains('chat-room-open'));
  await page.waitForFunction(()=>window.MDDChat.getTurns().length===1);
  await page.waitForFunction(()=>window.MDDChat.getTurns()[0].voice?.status==='ready');
  assert.equal(topicCalls.length,1); assert.equal(voiceCalls.length,1);
  assert.equal(voiceCalls[0].voice_id,sample.id); assert.equal(voiceCalls[0].speed,.85);
  assert.equal(await page.evaluate(()=>window.MDDChat.state.previewAudio),null);
  checks.push('Actual provider sample decodes/plays locally; selected accent/voice/speed reach the TTS request; starting stops preview.');
  await page.locator('#chat-stop-voice').click();
  await page.locator('#chat-topics').click();
  const beforeCancel=topicCalls.length;
  for(const width of [1440,768,390,320]) {
   await page.setViewportSize({width,height:width===320?640:900});
   await topic.click();
   const dialog=await page.locator('#chat-topic-setup').boundingBox();
   assert.ok(dialog.x>=0 && dialog.x+dialog.width<=width,`dialog bounds at ${width}`);
   assert.ok(dialog.y>=0 && dialog.y+dialog.height<=await page.evaluate(()=>innerHeight)+1,`dialog height at ${width}`);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   await page.locator('#topic-voice-search').fill('no-matching-voice');
   assert.equal(await page.locator('.topic-voice-row').count(),0);
   await page.locator('#topic-voice-search').fill('');
   if(width===390) {
    await page.locator('.topic-setup-body').evaluate(element=>{element.scrollTop=element.scrollHeight;});
    await page.screenshot({path:path.join(output,'voice-picker-mobile.png')});
   }
   if(width===320) { await page.locator('#topic-setup-close').click(); } else { await page.keyboard.press('Escape'); }
   await page.locator('#chat-topic-setup').waitFor({state:'hidden'});
   assert.equal(topicCalls.length,beforeCancel);
  }
  checks.push('Search, cancel, Escape and responsive layout pass at 320/390/768/1440 px without starting a topic.');
  await page.reload(); await page.waitForFunction(()=>window.MDDChat?.state.config);
  assert.equal(await page.evaluate(()=>window.MDDChat.state.voiceId),sample.id);
  assert.equal(await page.evaluate(()=>window.MDDChat.state.voiceSpeed),.85);
  // Test usable fallback when ai33 credentials are unavailable.
  await page.route('**/api/chat/config',route=>route.fulfill({json:{...readyConfig,ai33_available:false,default_voice_provider:'browser',voice_locales:[]}}));
  await page.reload(); await page.waitForFunction(()=>window.MDDChat?.state.config);
  await page.locator('.chat-topic-card').first().click();
  assert.equal(await page.locator('input[name=topic-voice][value=browser]').isChecked(),true);
  assert.equal(await page.locator('.topic-voice-choice input:not(:disabled)').count(),0);
  await page.locator('#topic-setup-auto-read').uncheck();
  await page.locator('#topic-setup-start').click();
  await page.waitForFunction(()=>window.MDDChat.state.inRoom);
  assert.equal(voiceCalls.length,1);
  checks.push('Voice/speed persist after reload; missing ai33 configuration falls back to browser voice without paid calls.');
  assert.deepEqual(errors,[]);
  const report={status:'passed',voice_count:config.english_voices.length,checks,errors,
   paid_tts_calls:0,scope:'Real local ai33 catalog and downloaded samples; model availability is a fixture. Topic generation and TTS responses are fixtures; no microphone or model inference.'};
  fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report,null,2));
 } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
