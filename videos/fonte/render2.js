// usage: node render2.js page.html test outdir t1,t2,...
//        node render2.js page.html full outdir fps startFrame endFrame
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const [, , page, mode, outdir, a, b, c] = process.argv;
  const br = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--allow-file-access-from-files'] });
  const p = await br.newPage({ viewport: { width: 720, height: 1280 } });
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + path.join(__dirname, page));
  await p.waitForFunction(() => window.renderAt, null, { timeout: 120000 });
  fs.mkdirSync(outdir, { recursive: true });
  let jobs;
  if (mode === 'test') jobs = a.split(',').map(t => [+t, 't_' + String(t).padStart(5, '0')]);
  else { const fps = +a; jobs = []; for (let i = +b; i < +c; i++) jobs.push([i / fps, 'f_' + String(i).padStart(5, '0')]); }
  for (let i = 0; i < jobs.length; i++) {
    await p.evaluate(t => renderAt(t), jobs[i][0]);
    await p.screenshot({ path: path.join(outdir, jobs[i][1] + '.jpg'), type: 'jpeg', quality: 92 });
    if (i % 200 === 0) console.log(page, 'frame', i, '/', jobs.length);
  }
  await br.close();
})();
