"""
Run this once every trading day BEFORE market open.
Upstox access tokens expire at 3:30 AM the following day, so this
needs to be re-run each morning (there's no free auto-refresh flow).

Usage:
    python auth/get_token.py
"""
import sys
import os
import json
import requests
from urllib.parse import urlparse, parse_qs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_login_url() -> str:
    return (
        "https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={config.UPSTOX_API_KEY}"
        f"&redirect_uri={config.UPSTOX_REDIRECT_URI}"
    )


def exchange_code_for_token(code: str) -> str:
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        data={
            "code": code,
            "client_id": config.UPSTOX_API_KEY,
            "client_secret": config.UPSTOX_API_SECRET,
            "redirect_uri": config.UPSTOX_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        headers={"accept": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def main():
    print("1. Open this URL in your browser and log in to Upstox:\n")
    print(get_login_url())
    print(
        "\n2. After login you'll land on your redirect_uri with a "
        "?code=... in the address bar (the page itself may show an error "
        "- that's fine, you only need the URL)."
    )
    pasted = input("\n3. Paste the FULL redirected URL (or just the code) here: ").strip()

    if "code=" in pasted:
        code = parse_qs(urlparse(pasted).query)["code"][0]
    else:
        code = pasted

    token = exchange_code_for_token(code)
    with open(config.TOKEN_FILE, "w") as f:
        json.dump({"access_token": token}, f)

    print(f"\n✅ Access token saved to {config.TOKEN_FILE}")
    print("Valid until 3:30 AM tonight/tomorrow. Now start main.py.")


if __name__ == "__main__":
    main()
