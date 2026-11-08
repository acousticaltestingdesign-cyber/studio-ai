"""Send one control-change probe to the studio desktop executor.

Loads STUDIO_DESKTOP_IP and STUDIO_DESKTOP_PORT from config.env beside this
file, posts a Blofeld control-change payload to /v1/execute, and prints the
bytes that were sent plus the HTTP status the receiver returned. The call
gives up after 2 seconds so a silent desktop cannot hold the process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

TIMEOUT_SECONDS = 2.0
EXECUTE_PATH = "/v1/execute"

PAYLOAD = {
    "target": "blofeld",
    "messages": [
        {
            "type": "control_change",
            "control": 74,
            "value": 90,
            "channel": 0,
        }
    ],
}


def main() -> int:
    env_path = Path(__file__).resolve().parent / "config.env"
    if not env_path.is_file():
        print(f"config missing path={env_path}")
        return 1

    load_dotenv(env_path, override=True)
    host = os.getenv("STUDIO_DESKTOP_IP", "").strip()
    port_text = os.getenv("STUDIO_DESKTOP_PORT", "").strip()
    if not host or not port_text.isdecimal():
        print("config missing STUDIO_DESKTOP_IP or STUDIO_DESKTOP_PORT")
        return 1

    url = f"http://{host}:{int(port_text)}{EXECUTE_PATH}"
    body = json.dumps(PAYLOAD, separators=(",", ":"), ensure_ascii=True)
    print(f"target {url}")
    print(f"payload {body}")

    try:
        response = requests.post(
            url,
            data=body.encode("ascii"),
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        print(f"error timeout seconds={TIMEOUT_SECONDS}")
        return 1
    except requests.RequestException as exc:
        print(f"error {exc.__class__.__name__}")
        return 1

    print(f"status {response.status_code}")
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
