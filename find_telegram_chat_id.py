"""
Find Telegram chat ids for the NSE bot (e.g. the group to add as a 2nd receiver).

    python find_telegram_chat_id.py

Before running: add the bot to the group and send any message there,
e.g. /start@<your_bot_name> (a /command reaches the bot even with group
privacy mode on). Reads TELEGRAM_BOT_TOKEN from .env - never prints it.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN")
if not token:
    raise SystemExit("TELEGRAM_BOT_TOKEN not found in .env")
me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=20).json()
print(f"Bot: @{me.get('result', {}).get('username', '?')}")
r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
if not r.get("ok"):
    raise SystemExit(f"Telegram error: {r.get('description')}  (a webhook set on this bot blocks getUpdates)")
chats = {}
for u in r.get("result", []):
    for k in ("message", "edited_message", "channel_post", "my_chat_member", "chat_member"):
        c = (u.get(k) or {}).get("chat")
        if c:
            chats[c["id"]] = c
if not chats:
    print("No chats seen yet. In the group send:  /start@" + me.get("result", {}).get("username", "your_bot") +
          "\nthen run this again (updates are kept by Telegram for 24 h).")
for cid, c in chats.items():
    name = c.get("title") or " ".join(filter(None, [c.get("first_name"), c.get("last_name")])) or c.get("username", "")
    kind = c.get("type")
    tag = "  <-- GROUP: put this in the GitHub secret TELEGRAM_GROUP_CHAT_ID" if kind in ("group", "supergroup") else ""
    print(f"{kind:<10} {cid:>16}  {name}{tag}")
