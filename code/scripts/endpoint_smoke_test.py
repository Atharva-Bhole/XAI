import json
import os
import tempfile
import time
from typing import Iterable

import requests

BASE_URL = "http://127.0.0.1:5000"
TIMEOUT = 120


def preview_response(resp: requests.Response) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            return json.dumps(resp.json(), ensure_ascii=True)[:240]
        except Exception:
            return (resp.text or "")[:240]
    if "application/pdf" in ctype:
        return f"<PDF {len(resp.content)} bytes>"
    text = (resp.text or "").strip()
    if text:
        return text[:240]
    return f"<{len(resp.content)} bytes; content-type={ctype or 'unknown'}>"


def run_check(
    session: requests.Session,
    results: list,
    name: str,
    method: str,
    path: str,
    expected_statuses: Iterable[int],
    **kwargs,
):
    url = f"{BASE_URL}{path}"
    expected = set(expected_statuses)
    try:
        resp = session.request(method=method, url=url, timeout=TIMEOUT, **kwargs)
        ok = resp.status_code in expected
        results.append(
            {
                "name": name,
                "ok": ok,
                "status": resp.status_code,
                "expected": sorted(expected),
                "preview": preview_response(resp),
            }
        )
        return resp
    except Exception as exc:
        results.append(
            {
                "name": name,
                "ok": False,
                "status": "EXCEPTION",
                "expected": sorted(expected),
                "preview": str(exc),
            }
        )
        return None


def create_test_image(path: str) -> None:
    # 1x1 PNG (black pixel)
    png_bytes = bytes.fromhex(
        "89504E470D0A1A0A"
        "0000000D4948445200000001000000010802000000907753DE"
        "0000000C4944415408D76360000000020001E221BC3300000000"
        "49454E44AE426082"
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def main() -> int:
    results = []
    session = requests.Session()

    ts = int(time.time())
    email = f"megha{ts}@example.com"
    password = "Pass@1234"

    # Public + auth flow
    run_check(session, results, "health", "GET", "/api/health", [200])
    run_check(session, results, "auth_me_unauth", "GET", "/api/auth/me", [401])
    run_check(
        session,
        results,
        "auth_register",
        "POST",
        "/api/auth/register",
        [201],
        json={
            "name": "Megha",
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )
    run_check(
        session,
        results,
        "auth_login",
        "POST",
        "/api/auth/login",
        [200],
        json={"email": email, "password": password},
    )
    run_check(session, results, "auth_me", "GET", "/api/auth/me", [200])

    # Protected dashboard
    run_check(session, results, "dashboard_stats", "GET", "/api/dashboard/stats", [200])

    # Analysis text (main happy path)
    text_resp = run_check(
        session,
        results,
        "analysis_text",
        "POST",
        "/api/analysis/text",
        [200],
        json={"text": "This product is amazing and very useful."},
    )
    analysis_id = None
    if text_resp is not None and text_resp.status_code == 200:
        try:
            analysis_id = text_resp.json().get("id")
        except Exception:
            analysis_id = None

    # Image endpoint (multipart success path)
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "sample.png")
        create_test_image(img_path)
        with open(img_path, "rb") as f:
            run_check(
                session,
                results,
                "analysis_image",
                "POST",
                "/api/analysis/image",
                [200],
                files={"image": ("sample.png", f, "image/png")},
            )

    # URL endpoint (public URL; some environments may return 422 if fetch blocked)
    run_check(
        session,
        results,
        "analysis_url",
        "POST",
        "/api/analysis/url",
        [200, 422],
        json={"url": "https://example.com"},
    )

    # Audio endpoint (validation path to confirm route behavior)
    run_check(
        session,
        results,
        "analysis_audio_no_file",
        "POST",
        "/api/analysis/audio",
        [400],
        files={},
    )

    # History and single analysis endpoints
    run_check(session, results, "analysis_history", "GET", "/api/analysis/history", [200])
    if analysis_id is not None:
        run_check(
            session,
            results,
            "analysis_get_by_id",
            "GET",
            f"/api/analysis/{analysis_id}",
            [200],
        )
        run_check(
            session,
            results,
            "analysis_report_download",
            "GET",
            f"/api/analysis/{analysis_id}/report",
            [200],
        )
    else:
        results.append(
            {
                "name": "analysis_get_by_id",
                "ok": False,
                "status": "SKIPPED",
                "expected": [200],
                "preview": "Skipped because analysis_text did not return an id.",
            }
        )
        results.append(
            {
                "name": "analysis_report_download",
                "ok": False,
                "status": "SKIPPED",
                "expected": [200],
                "preview": "Skipped because analysis_text did not return an id.",
            }
        )

    # Logout and post-logout protected check
    run_check(session, results, "auth_logout", "POST", "/api/auth/logout", [200])
    run_check(session, results, "dashboard_stats_post_logout", "GET", "/api/dashboard/stats", [401])

    # Print report
    passed = 0
    failed = 0
    print("\n=== X-Sense Endpoint Smoke Test ===")
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        if r["ok"]:
            passed += 1
        else:
            failed += 1
        print(
            f"[{mark}] {r['name']}: status={r['status']} expected={r['expected']}\n"
            f"       {r['preview']}"
        )

    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(results)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
