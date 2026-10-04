/* eslint-disable @typescript-eslint/no-require-imports */
/**
 * Focused cross-filter verification (dev utility):
 *   /analytics:  none -> May -> toggle -> May -> Fatal -> remove Fatal -> remove May
 *   /dashboard:  one click + chip clear.
 * Requires: backend :8000, `npm run build && npm start` on :3000.
 */
const puppeteer = require("puppeteer-core");

const BASE = "http://127.0.0.1:3000";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const out = {};

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--no-sandbox", "--disable-gpu", "--window-size=1440,900"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const consoleErrors = [];
  const dashRequests = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text().slice(0, 180)); });
  page.on("pageerror", (e) => consoleErrors.push("pageerror: " + e.message.slice(0, 180)));
  page.on("request", (r) => {
    const u = r.url();
    if (u.includes("/api/analytics/dashboard")) dashRequests.push(u.replace(/^https?:\/\/[^/]+/, ""));
  });

  /* ---------------------------------------------------------------- helpers */
  const chips = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('[data-testid="active-filters"] button[aria-label^="Remove"]')].map((b) =>
        b.getAttribute("aria-label").replace("Remove ", "").replace(" filter", ""),
      ),
    );
  const status = () =>
    page.evaluate(() => {
      const kpi = document.querySelector('[data-testid="kpi-cards"]');
      const txt = kpi ? kpi.innerText : "";
      const total = (txt.match(/total collisions\s*([\d,]+)/i) || [])[1];
      const sub = /Matching the active filters/i.test(txt);
      const bar = document.querySelector('[data-testid="active-filters"]');
      const donut = document.querySelector('[data-testid="chart-severity"]');
      const donutCaption = donut ? (donut.innerText.match(/([\d,]+) collisions match/i) || [])[1] : undefined;
      return {
        total,
        kpiShowsFilteredSub: sub,
        donutCaptionTotal: donutCaption,
        allRecords: !!bar && bar.innerText.includes("Showing all available records"),
        filteredBy: !!bar && bar.innerText.includes("Showing records filtered by:"),
      };
    });
  // visual signature of a chart: every bar/sector geometry + opacity (proves data changed)
  const sig = (testid) =>
    page.evaluate((id) => {
      const box = document.querySelector(`[data-testid="${id}"]`);
      if (!box) return null;
      return [...box.querySelectorAll("path,rect")]
        .map((el) => `${el.getAttribute("fill") || ""}|${el.getAttribute("d") || ""}|${el.getAttribute("fill-opacity") || "1"}`)
        .join(";");
    }, testid);
  const signatures = async () => {
    const ids = ["chart-severity", "chart-by-road-type", "chart-by-weather", "chart-by-lighting", "chart-by-speed-limit", "chart-by-month", "chart-by-hour"];
    const res = {};
    for (const id of ids) res[id] = await sig(id);
    return res;
  };
  // per-category emphasis: selected category full opacity, others muted (never hidden)
  const emphasisState = async (testid) =>
    page.evaluate((id) => {
      const box = document.querySelector(`[data-testid="${id}"]`);
      if (!box) return null;
      const agg = {};
      for (const el of box.querySelectorAll("path")) {
        const key = `fill-opacity=${el.getAttribute("fill-opacity")} opacity=${el.getAttribute("opacity")}`;
        agg[key] = (agg[key] || 0) + 1;
      }
      return agg;
    }, testid);
  // click category #catIndex of a vertical chart (bars share x per category; data is ordered)
  const clickCategory = async (testid, catIndex) => {
    await page.evaluate((id) => {
      const box = document.querySelector(`[data-testid="${id}"]`);
      if (box) box.scrollIntoView({ block: "center", behavior: "instant" });
    }, testid);
    await sleep(1200); // let any bar re-animation from a filter change settle first
    const ok = await page.evaluate(
      (id, index) => {
        const box = document.querySelector(`[data-testid="${id}"]`);
        if (!box) return "chart not found";
        const rects = [...box.querySelectorAll(".recharts-bar-rectangle path")]
          .map((el) => ({ el, r: el.getBoundingClientRect() }))
          .filter((o) => o.r.width > 0 && o.r.height > 0 && o.r.top >= 0 && o.r.bottom <= window.innerHeight);
        if (!rects.length) return "no visible bars";
        const xs = [...new Set(rects.map((o) => Math.round(o.r.left + o.r.width / 2)))].sort((a, b) => a - b);
        if (index >= xs.length) return `only ${xs.length} categories (wanted #${index})`;
        const targetX = xs[index];
        const hits = rects.filter((o) => Math.abs(Math.round(o.r.left + o.r.width / 2) - targetX) <= 1);
        const bar = hits.sort((a, b) => b.r.height - a.r.height)[0];
        window.__clickTarget = { x: bar.r.left + bar.r.width / 2, y: bar.r.top + bar.r.height / 2 };
        window.__underPointer = document.elementFromPoint(
          bar.r.left + bar.r.width / 2,
          bar.r.top + bar.r.height / 2,
        )?.tagName;
        return "ok";
      },
      testid,
      catIndex,
    );
    if (ok !== "ok") return ok;
    const pt = await page.evaluate(() => window.__clickTarget);
    await page.mouse.move(pt.x, pt.y);
    await page.mouse.move(pt.x + 1, pt.y);
    await page.mouse.click(pt.x, pt.y);
    return "clicked";
  };
  const clickFatalSlice = async () => {
    await page.evaluate(() => {
      const box = document.querySelector('[data-testid="chart-severity"]');
      if (box) box.scrollIntoView({ block: "center", behavior: "instant" });
    });
    await sleep(1200); // donut re-animates after each filter change
    const ok = await page.evaluate(() => {
      const box = document.querySelector('[data-testid="chart-severity"]');
      if (!box) return "donut not found";
      const sector = [...box.querySelectorAll(".recharts-sector")].find(
        (el) => (el.getAttribute("fill") || "").toLowerCase() === "#ef4444",
      );
      if (!sector) return "Fatal sector not found";
      const r = sector.getBoundingClientRect();
      window.__clickTarget = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      return "ok";
    });
    if (ok !== "ok") return ok;
    const pt = await page.evaluate(() => window.__clickTarget);
    await page.mouse.move(pt.x, pt.y);
    await page.mouse.click(pt.x, pt.y);
    return "clicked";
  };
  const clickChip = async (label) => {
    await page.evaluate((l) => {
      const btn = [...document.querySelectorAll('[data-testid="active-filters"] button[aria-label^="Remove"]')]
        .find((b) => b.getAttribute("aria-label") === `Remove ${l} filter`);
      if (btn) btn.click();
    }, label);
  };
  const waitFor = async (fn, ms = 12000, every = 200) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await fn()) return true;
      await sleep(every);
    }
    return false;
  };

  /* ---------------------------------------------------------------- 1. /analytics baseline */
  await page.goto(BASE + "/analytics", { waitUntil: "domcontentloaded", timeout: 30000 });
  const baselineLoaded = await waitFor(async () => (await status()).total === "513,801");
  await sleep(1800); // let entry/chart animations settle before snapshotting signatures
  const before = { status: await status(), chips: await chips(), sigs: await signatures() };
  out.step1_baseline = { loaded: baselineLoaded, ...before.status, chips: before.chips };

  // hover only -> tooltip may appear, but NO filter may be applied
  await page.evaluate(() => {
    const box = document.querySelector('[data-testid="chart-by-month"]');
    if (box) box.scrollIntoView({ block: "center", behavior: "instant" });
  });
  await sleep(400);
  const hoverPt = await page.evaluate(() => {
    const box = document.querySelector('[data-testid="chart-by-month"]');
    const path = [...box.querySelectorAll(".recharts-bar-rectangle path")].find(
      (p) => p.getBoundingClientRect().height > 4,
    );
    const r = path.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });
  await page.mouse.move(hoverPt.x, hoverPt.y);
  await page.mouse.move(hoverPt.x + 2, hoverPt.y);
  await sleep(700);
  const tooltip = await page.evaluate(() => {
    const all = [...document.querySelectorAll('[class*="tooltip"]')];
    const visible = all.filter((el) => {
      const cs = getComputedStyle(el);
      return cs.visibility !== "hidden" && cs.display !== "none" && el.innerText;
    });
    return {
      tooltip_classes: all.slice(0, 6).map((el) => el.className && String(el.className).slice(0, 60)),
      visible_text: visible[0] ? visible[0].innerText.replace(/\s+/g, " ").slice(0, 140) : null,
    };
  });
  out.step2_hover_no_filter = { tooltip, chips: await chips(), ...(await status()) };

  /* ---------------------------------------------------------------- 2. click May */
  const mayClick = await clickCategory("chart-by-month", 4);
  const mayApplied = await waitFor(async () => {
    const st = await status();
    return st.total === "45,204";
  });
  const afterMay = { status: await status(), chips: await chips(), sigs: await signatures() };
  out.step3_click_may = {
    click: mayClick,
    applied: mayApplied,
    ...afterMay.status,
    chips: afterMay.chips,
    severity_changed: afterMay.sigs["chart-severity"] !== before.sigs["chart-severity"],
    road_changed: afterMay.sigs["chart-by-road-type"] !== before.sigs["chart-by-road-type"],
    weather_changed: afterMay.sigs["chart-by-weather"] !== before.sigs["chart-by-weather"],
    lighting_changed: afterMay.sigs["chart-by-lighting"] !== before.sigs["chart-by-lighting"],
    speed_changed: afterMay.sigs["chart-by-speed-limit"] !== before.sigs["chart-by-speed-limit"],
    month_chart_still_has_12_bars: await page.evaluate(() => {
      const box = document.querySelector('[data-testid="chart-by-month"]');
      const xs = new Set(
        [...box.querySelectorAll(".recharts-bar-rectangle path")].map((p) => Math.round(p.getBoundingClientRect().left)),
      );
      return xs.size;
    }),
    month_bar_emphasis: await emphasisState("chart-by-month"),
    last_request: dashRequests[dashRequests.length - 1],
  };

  /* ---------------------------------------------------------------- 3. second click on May toggles off */
  await clickCategory("chart-by-month", 4);
  const toggledOff = await waitFor(async () => (await status()).total === "513,801");
  out.step4_second_click_toggles_off = { applied: toggledOff, chips: await chips(), ...(await status()) };

  // select May again for the rest of the sequence
  await clickCategory("chart-by-month", 4);
  await waitFor(async () => (await status()).total === "45,204");

  /* ---------------------------------------------------------------- 4. click Fatal */
  let fatalClick = await clickFatalSlice();
  let fatalApplied = await waitFor(async () => (await status()).total === "635", 8000);
  if (!fatalApplied) {
    fatalClick = `${fatalClick} + retry:${await clickFatalSlice()}`;
    fatalApplied = await waitFor(async () => (await status()).total === "635", 8000);
  }
  const afterFatal = { status: await status(), chips: await chips(), sigs: await signatures() };
  out.step5_click_fatal = {
    click: fatalClick,
    applied: fatalApplied,
    ...afterFatal.status,
    chips: afterFatal.chips,
    weather_changed_again: afterFatal.sigs["chart-by-weather"] !== afterMay.sigs["chart-by-weather"],
    lighting_changed_again: afterFatal.sigs["chart-by-lighting"] !== afterMay.sigs["chart-by-lighting"],
    road_changed_again: afterFatal.sigs["chart-by-road-type"] !== afterMay.sigs["chart-by-road-type"],
    donut_emphasis: await emphasisState("chart-severity"),
    last_request: dashRequests[dashRequests.length - 1],
  };

  /* ---------------------------------------------------------------- 5. remove Fatal chip */
  await clickChip("Fatal");
  const fatalRemoved = await waitFor(async () => (await status()).total === "45,204");
  out.step6_remove_fatal_chip = { applied: fatalRemoved, chips: await chips(), ...(await status()) };

  /* ---------------------------------------------------------------- 6. remove May chip */
  await clickChip("May");
  const allRestored = await waitFor(async () => {
    const st = await status();
    return st.total === "513,801" && st.allRecords;
  });
  await sleep(1600); // let bar re-animations settle before comparing geometry
  const afterClear = { status: await status(), chips: await chips(), sigs: await signatures() };
  out.step7_remove_may_chip = {
    restored: allRestored,
    ...afterClear.status,
    chips: afterClear.chips,
    all_charts_restored:
      afterClear.sigs["chart-by-weather"] === before.sigs["chart-by-weather"] &&
      afterClear.sigs["chart-severity"] === before.sigs["chart-severity"] &&
      afterClear.sigs["chart-by-lighting"] === before.sigs["chart-by-lighting"],
  };

  /* ---------------------------------------------------------------- 7. /dashboard spot check */
  await page.goto(BASE + "/dashboard", { waitUntil: "domcontentloaded", timeout: 30000 });
  const dashLoaded = await waitFor(async () => (await status()).total === "513,801");
  await sleep(1500);
  const dashBefore = await status();
  const dashClick = await clickCategory("chart-by-month", 4);
  const dashApplied = await waitFor(async () => (await status()).total === "45,204");
  const dashAfter = { status: await status(), chips: await chips() };
  await clickChip("May");
  const dashCleared = await waitFor(async () => (await status()).total === "513,801");
  out.step8_dashboard_page = {
    loaded: dashLoaded,
    click: dashClick,
    before: dashBefore,
    may_applied: dashApplied,
    after_may: dashAfter.status,
    chips_after_may: dashAfter.chips,
    cleared: dashCleared,
    chips_after_clear: await chips(),
  };

  /* ---------------------------------------------------------------- 9. dropdowns share the state */
  await page.goto(BASE + "/analytics", { waitUntil: "domcontentloaded", timeout: 30000 });
  await waitFor(async () => (await status()).total === "513,801");
  await sleep(1200);
  await page.select("#filter-severity", "Fatal");
  const ddApplied = await waitFor(async () => (await status()).total === "7,553");
  const ddState = { status: await status(), chips: await chips() };
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Clear all filters");
    if (btn) btn.click();
  });
  const ddCleared = await waitFor(async () => {
    const st = await status();
    return st.total === "513,801" && st.allRecords;
  });
  out.step9_dropdown_integration = {
    severity_dropdown_applied: ddApplied,
    ...ddState.status,
    chips: ddState.chips,
    cleared: ddCleared,
    chips_after: await chips(),
  };

  out.requests = dashRequests;
  out.console_errors = consoleErrors.slice(0, 10);
  await browser.close();
  console.log(JSON.stringify(out, null, 1));
})().catch((e) => { console.error("FATAL", e); process.exit(1); });
