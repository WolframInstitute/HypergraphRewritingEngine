// The package in a browser: serves js/ with the cross-origin isolation headers (COOP and COEP)
// the threaded module needs, opens test/browser/index.html in headless Chromium and checks the
// counts it reports from the page's thread and from Web Workers. Needs playwright-core and its
// Chromium: PLAYWRIGHT_CORE names the playwright-core directory.
//   PLAYWRIGHT_CORE=<dir>/node_modules/playwright-core node test/browser.mjs
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const TYPES = { ".mjs": "text/javascript", ".js": "text/javascript", ".html": "text/html", ".wasm": "application/wasm" };
const core = process.env.PLAYWRIGHT_CORE;
if (!core) { console.error("set PLAYWRIGHT_CORE to the playwright-core directory"); process.exit(2); }
const { chromium } = await import(pathToFileURL(join(core, "index.mjs")).href);

const server = createServer(async (req, res) => {
    const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname)));
    if (!path.startsWith(ROOT)) { res.writeHead(403).end(); return; }
    try {
        const body = await readFile(path);
        res.writeHead(200, { "Content-Type": TYPES[extname(path)] ?? "application/octet-stream",
            "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" });
        res.end(body);
    } catch {
        res.writeHead(404).end();
    }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/test/browser/index.html`;

const browser = await chromium.launch();
let failed = false;
try {
    const page = await browser.newPage();
    page.on("console", (m) => { if (m.type() === "error") console.error("console:", m.text()); });
    await page.goto(url);
    await page.waitForFunction(() => window.hgResults !== undefined, null, { timeout: 120000 });
    const r = await page.evaluate(() => window.hgResults);
    console.log(JSON.stringify(r, null, 1));
    const check = (name, x, threaded) => {
        const ok = x && !x.error && x.threaded === threaded && x.numStates === 10 && x.numEvents === 9;
        console.log(`${ok ? "PASS" : "FAIL"} ${name}`);
        failed ||= !ok;
    };
    check("single-threaded module on the page's thread", r.page, false);
    check("single-threaded module in a Web Worker", r.worker, false);
    check("threaded module in a Web Worker", r.workerThreads, true);
    const big = r.worker.bigStates === r.workerThreads.bigStates && r.worker.bigEvents === r.workerThreads.bigEvents;
    console.log(`${big ? "PASS" : "FAIL"} both modules give the same counts on a 5-step Full evolution`);
    failed ||= !big || r.crossOriginIsolated !== true;
} finally {
    await browser.close();
    server.close();
}
process.exit(failed ? 1 : 0);
