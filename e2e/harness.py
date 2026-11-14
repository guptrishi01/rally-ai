#!/usr/bin/env python3
"""Drives the running dashboard with Playwright and reports what it saw.

The behaviour half of the test suite. `pytest` covers the pipelines and the
API, but nothing in Python can tell you the page throws on load, a tab
renders blank, or the layout scrolls sideways on a phone - only opening it
can. This opens it, in every viewport, and reports console errors, uncaught
exceptions, failed requests and assertion results as JSON.

Not run by `pytest`: it needs a built frontend and a live server, so it is
its own step. `scripts/e2e_check.py` is the entry point that rebuilds,
restarts and runs it; this module is just the driver.

Deliberately framework-agnostic - it takes a URL and a config, not a build
system - so the harness survives whatever the frontend becomes.

Config shape: see e2e/dashboard.json and _check_assertions below.

Usage:
    python e2e/harness.py run --config e2e/dashboard.json --out e2e/output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - environment-dependent
    sync_playwright = None

# Noise that is not the app's fault and would otherwise fail every run.
IGNORED_CONSOLE = (
    "favicon.ico",
    "Download the React DevTools",
    "[vite] connected",
)


def _ignorable(text: str) -> bool:
    """Whether a console message is known noise rather than a real error.

    Args:
        text: The console message.

    Returns:
        True if it should not count against the run.
    """
    return any(pattern in text for pattern in IGNORED_CONSOLE)


def _apply_actions(page, actions: list[dict]) -> list[str]:
    """Performs a step's actions in order.

    Args:
        page: The Playwright page.
        actions: Action dicts - {"click": sel}, {"fill": [sel, value]},
            {"wait": ms}, {"press": [sel, key]}.

    Returns:
        Messages for any action that could not be performed.
    """
    problems: list[str] = []
    for action in actions:
        try:
            if "click" in action:
                page.click(action["click"], timeout=5000)
            elif "fill" in action:
                selector, value = action["fill"]
                page.fill(selector, value, timeout=5000)
            elif "press" in action:
                selector, key = action["press"]
                page.press(selector, key, timeout=5000)
            elif "wait" in action:
                page.wait_for_timeout(int(action["wait"]))
            else:
                problems.append(f"unknown action: {action!r}")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            problems.append(f"action {action!r} failed: {type(exc).__name__}: {exc}")
    return problems


def _check_assertions(page, spec: dict) -> list[str]:
    """Evaluates a step's assertions against the live page.

    Args:
        page: The Playwright page.
        spec: The step's "assert" block.

    Returns:
        One message per failed assertion; empty if all held.
    """
    failures: list[str] = []

    for selector in spec.get("selector_present", []):
        if page.locator(selector).count() == 0:
            failures.append(f"missing selector: {selector}")

    for selector in spec.get("selector_absent", []):
        if page.locator(selector).count() > 0:
            failures.append(f"selector should not be present: {selector}")

    body = page.inner_text("body") if page.locator("body").count() else ""
    for text in spec.get("text_present", []):
        if text not in body:
            failures.append(f"missing text: {text!r}")
    for text in spec.get("text_absent", []):
        if text in body:
            failures.append(f"text should not appear: {text!r}")

    if spec.get("no_horizontal_scroll"):
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        if overflow > 1:  # 1px of rounding is not a layout bug
            failures.append(f"horizontal scroll: {overflow}px wider than the viewport")

    return failures


def run(config: dict, out_dir: Path, headed: bool = False) -> dict:
    """Runs every step in every viewport and reports what happened.

    Args:
        config: The parsed config (see module docstring).
        out_dir: Where screenshots are written.
        headed: Show the browser, for watching a failure happen.

    Returns:
        A report dict; its "summary" is what the ledger judges.
    """
    if sync_playwright is None:
        return {
            "summary": {"ok": False, "failures": 1, "console_errors": 0, "steps": 0},
            "error": (
                "playwright is not installed "
                "(pip install playwright && playwright install chromium)"
            ),
            "results": [],
        }

    base_url = config["base_url"].rstrip("/")
    viewports = config.get(
        "viewports", [{"name": "desktop", "width": 1440, "height": 900}]
    )
    steps = config.get("steps", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        for viewport in viewports:
            context = browser.new_context(
                viewport={"width": viewport["width"], "height": viewport["height"]}
            )
            page = context.new_page()

            console_errors: list[str] = []
            page.on(
                "console",
                lambda m: console_errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" and not _ignorable(m.text)
                else None,
            )
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
            failed_requests: list[str] = []
            page.on(
                "requestfailed",
                lambda r: failed_requests.append(f"{r.method} {r.url}")
                if not _ignorable(r.url)
                else None,
            )

            for step in steps:
                before = len(console_errors)
                name = f"{step['name']}-{viewport['name']}"
                problems: list[str] = []
                try:
                    page.goto(base_url + step.get("goto", "/"), wait_until="networkidle",
                              timeout=30000)
                    page.wait_for_timeout(step.get("settle", 400))
                    problems += _apply_actions(page, step.get("actions", []))
                    problems += _check_assertions(page, step.get("assert", {}))
                except Exception as exc:  # noqa: BLE001 - becomes a reported failure
                    problems.append(f"{type(exc).__name__}: {exc}")

                shot = out_dir / f"{name}.png"
                try:
                    page.screenshot(path=str(shot), full_page=True)
                except Exception:  # noqa: BLE001 - a missing screenshot isn't the finding
                    shot = None

                step_console = console_errors[before:]
                results.append(
                    {
                        "step": step["name"],
                        "viewport": viewport["name"],
                        "url": base_url + step.get("goto", "/"),
                        "ok": not problems and not step_console,
                        "problems": problems,
                        "console_errors": step_console,
                        "screenshot": str(shot) if shot else None,
                    }
                )

            if failed_requests:
                results.append(
                    {
                        "step": "network",
                        "viewport": viewport["name"],
                        "ok": False,
                        "problems": [f"request failed: {r}" for r in failed_requests],
                        "console_errors": [],
                        "screenshot": None,
                    }
                )
            context.close()
        browser.close()

    failures = sum(1 for r in results if not r["ok"])
    console_total = sum(len(r["console_errors"]) for r in results)
    return {
        "summary": {
            "ok": failures == 0,
            "steps": len(results),
            "failures": failures,
            "console_errors": console_total,
        },
        "results": results,
    }


def cmd_run(args) -> int:
    """Runs the config and prints a human log then the JSON report."""
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run(config, Path(args.out), headed=args.headed)

    for result in report["results"]:
        mark = "ok  " if result["ok"] else "FAIL"
        print(f"[{mark}] {result['step']} @ {result['viewport']}")
        for problem in result["problems"]:
            print(f"        {problem}")
        for error in result["console_errors"]:
            print(f"        {error}")
    if "error" in report:
        print(report["error"])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    report_path = Path(args.out) / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {report_path}")
    # Printed last so a ledger `json:` expectation can read it, and wrapped
    # so the path is the same one report.json uses - a claim written as
    # `summary.console_errors` must mean the same thing in both places, or
    # it silently fails for the wrong reason on a healthy app.
    print(json.dumps({"summary": report["summary"]}))
    return 0 if report["summary"]["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status; nonzero when the app misbehaved.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="drive the app and report")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default=".shipgate/e2e")
    p_run.add_argument("--headed", action="store_true")
    p_run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
