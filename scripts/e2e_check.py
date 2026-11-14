"""Runs the dashboard's behaviour checks against a guaranteed-fresh build.

The entry point for the behaviour half of the suite: rebuild the frontend,
make sure a server is up, prove it is actually serving the build we just
made, then drive it with e2e/harness.py.

The freshness proof is the part that earns its keep. The harness tests what
is actually running, not the source tree, and a stale run invalidates the
result in both directions - reporting failures you already fixed, or passing
changes you already broke.

Two different things can go stale here, and only one is obvious:

- **dist/ older than src/.** The real risk for this app. Flask serves
  frontend/dist/ as static files, so editing a .tsx and forgetting to build
  means the browser is handed the previous UI with no sign anything is
  wrong. Checked by comparing modification times.
- **served bundle != built bundle.** Cannot normally happen while Flask
  serves dist/ straight off disk, but would the moment anything caches or a
  different server is pointed at the URL. Cheap to check, so it is checked.

Usage:
    python scripts/e2e_check.py                 # rebuild, serve, check
    python scripts/e2e_check.py --skip-build    # reuse the current dist/
    python scripts/e2e_check.py --restart       # replace a running server
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND = _REPO_ROOT / "frontend"
_DIST_INDEX = _FRONTEND / "dist" / "index.html"
_CONFIG = _REPO_ROOT / "e2e" / "dashboard.json"
_HARNESS = _REPO_ROOT / "e2e" / "harness.py"
_OUT = _REPO_ROOT / "e2e" / "output"
_URL = "http://127.0.0.1:5000"
_BUNDLE = re.compile(r"assets/index-[A-Za-z0-9_-]+\.js")


def built_bundle() -> str:
    """The hashed entry bundle the current build produced.

    Returns:
        The bundle's filename, e.g. "assets/index-K_AAnZmh.js".

    Raises:
        SystemExit: If dist/ is missing or has no bundle reference, which
            means there is nothing to test.
    """
    if not _DIST_INDEX.exists():
        raise SystemExit(f"No build found at {_DIST_INDEX}. Run: cd frontend && npm run build")
    match = _BUNDLE.search(_DIST_INDEX.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"Could not find a bundle reference in {_DIST_INDEX}")
    return match.group(0)


def stale_sources() -> list[Path]:
    """Frontend sources modified more recently than the build.

    The failure this catches is mundane and silent: edit a component, skip
    the build, and the served UI is the previous one. Nothing errors - the
    run simply reports on code that is not the code under test.

    Returns:
        Up to a few offending paths, newest first; empty when the build is
        current.
    """
    if not _DIST_INDEX.exists():
        return []
    built_at = _DIST_INDEX.stat().st_mtime
    sources = [
        path
        for path in (_FRONTEND / "src").rglob("*")
        if path.is_file() and path.stat().st_mtime > built_at
    ]
    index_html = _FRONTEND / "index.html"
    if index_html.exists() and index_html.stat().st_mtime > built_at:
        sources.append(index_html)
    return sorted(sources, key=lambda p: p.stat().st_mtime, reverse=True)[:5]


def served_bundle(timeout: float = 5.0) -> str | None:
    """The bundle the running server is actually serving, if it is up.

    Args:
        timeout: Seconds to wait for a response.

    Returns:
        The served bundle's filename, or None if nothing is listening.
    """
    try:
        with urllib.request.urlopen(_URL, timeout=timeout) as response:
            match = _BUNDLE.search(response.read().decode("utf-8", "replace"))
            return match.group(0) if match else ""
    except (urllib.error.URLError, OSError):
        return None


def rebuild() -> None:
    """Rebuilds the frontend, failing loudly if it does not compile.

    Raises:
        SystemExit: If the build fails - there is no point testing a build
            that does not exist.
    """
    print("building frontend...")
    result = subprocess.run(
        "npm run build", shell=True, cwd=_FRONTEND, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit("frontend build failed")
    print("  built")


def start_server() -> subprocess.Popen:
    """Starts the Flask app and waits for it to answer.

    Returns:
        The server process, for the caller to terminate.

    Raises:
        SystemExit: If it never came up.
    """
    print("starting server...")
    process = subprocess.Popen(
        [sys.executable, str(_REPO_ROOT / "webapp" / "app.py")],
        cwd=_REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        time.sleep(0.5)
        if served_bundle(timeout=2) is not None:
            print("  up")
            return process
    process.terminate()
    raise SystemExit("server did not start")


def stop_server(process: subprocess.Popen) -> None:
    """Stops a server this script started.

    Args:
        process: The process returned by start_server.
    """
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        The harness's exit status - nonzero when the dashboard misbehaved.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-build", action="store_true", help="reuse the current dist/")
    parser.add_argument("--restart", action="store_true", help="replace a running server")
    args = parser.parse_args(argv)

    if not args.skip_build:
        rebuild()
    expected = built_bundle()

    stale = stale_sources()
    if stale:
        listing = "\n".join(f"  {path.relative_to(_REPO_ROOT)}" for path in stale)
        raise SystemExit(
            "Stale build: these sources are newer than frontend/dist/.\n"
            f"{listing}\n"
            "Re-run without --skip-build, or build the frontend first."
        )

    owned: subprocess.Popen | None = None
    running = served_bundle()
    if running is not None and args.restart:
        raise SystemExit(
            "A server is already running. Stop it first, then re-run without --restart "
            "(this script will not kill a process it did not start)."
        )
    if running is None:
        owned = start_server()
        running = served_bundle()

    # The whole point: prove we are testing what we just built.
    if running != expected:
        if owned:
            stop_server(owned)
        raise SystemExit(
            f"Stale server: it serves {running or '(nothing)'} but the build is {expected}.\n"
            "Restart the server so it picks up the new build, then re-run."
        )
    print(f"serving the current build ({expected})\n")

    try:
        result = subprocess.run(
            [sys.executable, str(_HARNESS), "run", "--config", str(_CONFIG), "--out", str(_OUT)],
            cwd=_REPO_ROOT,
        )
    finally:
        if owned:
            stop_server(owned)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
