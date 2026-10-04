/* eslint-disable @typescript-eslint/no-require-imports */
/**
 * End-to-end UI check (development utility, not part of the app):
 * loads every route in headless Chrome, collects console/page/request errors,
 * verifies the dashboard shows real API numbers and runs a real prediction
 * through the Prediction Studio form.
 *
 * Requires: backend on :8000, `npm run build && npm start` on :3000.
 *   node e2e-check.cjs
 */
const puppeteer = require("puppeteer-core");

const BASE = "http://127.0.0.1:3000";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const ROUTES = ["/", "/dashboard", "/predict", "/analytics", "/model", "/insights", "/map", "/methodology", "/about"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--disable-gpu", "--window-size=1440,900"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const report = {};
  let errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(`[console] ${m.text()}`); });
  page.on("pageerror", (e) => errors.push(`[pageerror] ${e.message}`));
  // The basemap must be plain OpenStreetMap: any CARTO/cartocdn request is a failure.
  page.on("request", (r) => {
    if (/carto/i.test(r.url())) errors.push(`[carto-request] ${r.url()}`);
  });
  page.on("requestfailed", (r) => {
    const u = r.url();
    if (!u.includes("openstreetmap")) {
      errors.push(`[requestfailed] ${u} ${r.failure() && r.failure().errorText}`);
    }
  });

  // ---------------------------------------------------- every route renders
  for (const route of ROUTES) {
    errors = [];
    try {
      await page.goto(BASE + route, { waitUntil: "domcontentloaded", timeout: 45000 });
      await sleep(3000);
      const text = await page.evaluate(() => document.body.innerText);
      report[route] = {
        chars: text.length,
        nan: text.includes("NaN"),
        unexpected_undefined: /\bundefined\b/.test(text),
        errors: [...errors],
      };
    } catch (e) {
      report[route] = { fatal: String(e), errors: [...errors] };
    }
  }

  // ---------------------------------------------------- dashboard shows real totals
  errors = [];
  await page.goto(BASE + "/dashboard", { waitUntil: "domcontentloaded" });
  await sleep(3500);
  const dashText = await page.evaluate(() => document.body.innerText);
  report.dashboard_checks = {
    shows_total_513801: dashText.includes("513,801"),
    shows_fatal_share: dashText.includes("1.47") || dashText.includes("1.46"),
    shows_model_name: /Logistic Regression/i.test(dashText),
    errors: [...errors],
  };

  // ---------------------------------------------------- real prediction via the form
  errors = [];
  await page.goto(BASE + "/predict", { waitUntil: "domcontentloaded" });
  await sleep(2500);
  const select = async (id, value) => { await page.select(`#${id}`, value); await sleep(120); };
  const type = async (id, value) => { await page.click(`#${id}`, { clickCount: 3 }); await page.type(`#${id}`, value, { delay: 20 }); await sleep(80); };
  try {
    await select("f-road_type", "6");
    await select("f-junction_detail", "13");
    await select("f-light_conditions", "4");
    await select("f-weather_conditions", "2");
    await select("f-road_surface_conditions", "2");
    await select("f-urban_or_rural_area", "2");
    await type("f-speed_limit", "60");
    await type("f-number_of_vehicles", "2");
    await type("f-hour", "21");
    await select("f-day_of_week", "5");
    await page.click('button[type="submit"]');
    await page.waitForFunction(
      () => document.body.innerText.toLowerCase().includes("predicted severity"),
      { timeout: 30000 },
    );
    const resultText = await page.evaluate(() => document.body.innerText);
    const lower = resultText.toLowerCase();
    const pcts = [...resultText.matchAll(/(\d+\.\d{2})%/g)].map((m) => parseFloat(m[1]));
    const probSum = pcts.slice(0, 3).reduce((a, b) => a + b, 0);
    report.prediction_check = {
      succeeded: true,
      first_three_pct_sum: Number(probSum.toFixed(2)),
      has_disclaimer: lower.includes("should not be treated as a definitive"),
      has_confidence: lower.includes("confidence"),
      errors: [...errors],
    };
  } catch (e) {
    report.prediction_check = { succeeded: false, error: String(e), errors: [...errors] };
  }

  // ---------------------------------------------------- invalid input is rejected
  errors = [];
  await page.goto(BASE + "/predict", { waitUntil: "domcontentloaded" });
  await sleep(2000);
  try {
    await type("f-hour", "99"); // out of range -> local validation
    await page.click('button[type="submit"]');
    await sleep(900);
    const t = await page.evaluate(() => document.body.innerText);
    report.invalid_input_check = { rejected: /Max 23|Must be/.test(t), errors: [...errors] };
  } catch (e) {
    report.invalid_input_check = { rejected: false, error: String(e), errors: [...errors] };
  }

  await browser.close();
  console.log(JSON.stringify(report, null, 1));
  const failed =
    Object.values(report).some((v) => v && (v.fatal || (v.errors && v.errors.length))) ||
    !report.dashboard_checks?.shows_total_513801 ||
    !report.prediction_check?.succeeded ||
    !report.invalid_input_check?.rejected;
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
