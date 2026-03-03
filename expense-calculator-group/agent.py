"""
Receipt / Expense Calculator agent for ASI-One and Agentverse.
- Accepts receipt photo: extracts line items via OpenAI Vision.
- Or add items manually. Then poll (who brought what) and show fair split.
"""
from __future__ import annotations

import base64
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    MetadataContent,
    ResourceContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.storage import ExternalStorage
from dotenv import load_dotenv

from expense_logic import (
    Receipt,
    ReceiptItem,
    parse_item_selection,
    format_split_result,
    format_split_summary,
    format_split_summary_table,
    format_split_full_table,
)
from receipt_vision import extract_items_from_receipt_image

load_dotenv()

# Storage keys
RECEIPT_STATE = "expense_receipt_state"
RECEIPT_ITEMS = "expense_receipt_items"
RECEIPT_SELECTIONS = "expense_receipt_selections"
RECEIPT_NAMES = "expense_receipt_names"
RECEIPT_NAMES_USER_SET = "expense_receipt_names_user_set"  # senders who said "I'm X" (override profile)
RECEIPT_PARTICIPANTS = "expense_receipt_participants"  # senders seen in current polling round
RECEIPT_EXPECTED_COUNT = "expense_receipt_expected_count"  # optional total participants in group
RECEIPT_PAYER_SENDER = "expense_receipt_payer_sender"
RECEIPT_PAYER_NAME = "expense_receipt_payer_name"

STORAGE_URL = os.getenv("AGENTVERSE_URL", "https://agentverse.ai") + "/v1/storage"

agent = Agent(
    name="ReceiptCalculator",
    seed=os.getenv("AGENT_SEED", "receipt-calculator-seed"),
    mailbox=True,
    port=8004,
    publish_agent_details=True,
)

chat_proto = Protocol(spec=chat_protocol_spec)


def text_msg(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


def _get_uri_from_resource(res: Any) -> str | None:
    """Get URI from a resource that may be a dict or a Resource model."""
    if res is None:
        return None
    if isinstance(res, dict):
        return res.get("uri") or res.get("url")
    return getattr(res, "uri", None) or getattr(res, "url", None)


def _download_image_bytes(
    ctx: Context,
    resource_id: str,
    resource_list: list[Any] | None,
) -> bytes | None:
    """Download image bytes from storage or from resource URIs. Returns None on failure."""
    content_bytes: bytes | None = None
    if resource_id:
        try:
            ctx.logger.info(f"Downloading resource via storage: {resource_id}")
            storage = ExternalStorage(
                identity=ctx.agent.identity,
                storage_url=STORAGE_URL,
            )
            stored = storage.download(str(resource_id))
            content_bytes = base64.b64decode(stored.get("contents", ""))
        except Exception as exc:
            ctx.logger.info(f"Storage download failed: {exc}, trying URI fallback...")
    if not content_bytes and resource_list:
        for res in resource_list:
            uri = _get_uri_from_resource(res)
            if uri:
                try:
                    response = httpx.get(uri, timeout=60)
                    response.raise_for_status()
                    content_bytes = response.content
                    ctx.logger.info("URI fallback download succeeded")
                    break
                except Exception as e:
                    ctx.logger.warning(f"URI download failed: {e}")
    if not content_bytes:
        ctx.logger.warning("Could not download image: storage and URI fallback failed")
        return None

    if not content_bytes or len(content_bytes) < 100:
        ctx.logger.warning("Downloaded content too small or empty to be a valid image")
        return None
    # Basic image check
    if content_bytes.startswith(b"\xff\xd8\xff") or content_bytes.startswith(b"\x89PNG"):
        return content_bytes
    if content_bytes.startswith(b"GIF") or (content_bytes.startswith(b"RIFF") and b"WEBP" in content_bytes[:12]):
        return content_bytes
    # Allow unknown as image (e.g. heic or other)
    return content_bytes


def download_image_resource(ctx: Context, item: ResourceContent) -> bytes | None:
    """Download image bytes from Agentverse storage or URI. Returns None on failure."""
    resources = item.resource if isinstance(item.resource, list) else [item.resource]
    return _download_image_bytes(ctx, str(item.resource_id), resources)


def _load_receipt(ctx: Context) -> tuple[str, Receipt, dict[str, list[int]]]:
    state = ctx.storage.get(RECEIPT_STATE) or "draft"
    items_data = ctx.storage.get(RECEIPT_ITEMS) or []
    selections = ctx.storage.get(RECEIPT_SELECTIONS) or {}
    receipt = Receipt()
    for d in items_data:
        receipt.items.append(
            ReceiptItem(index=d["index"], name=d["name"], price=Decimal(str(d["price"])))
        )
    return state, receipt, selections


def _save_receipt(ctx: Context, state: str, receipt: Receipt, selections: dict[str, list[int]]) -> None:
    ctx.storage.set(RECEIPT_STATE, state)
    ctx.storage.set(
        RECEIPT_ITEMS,
        [{"index": i.index, "name": i.name, "price": str(i.price)} for i in receipt.items],
    )
    ctx.storage.set(RECEIPT_SELECTIONS, selections)


def _display_name_from_metadata(meta: dict) -> str | None:
    """Extract display name from ASI-One / chat metadata (profile can be object or string)."""
    if not meta:
        return None
    # Direct string fields
    for key in ("display_name", "name", "username", "displayName", "userName"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:50]
    # Nested: profile or user_preferences or user_profile (often objects with name/display_name)
    for key in ("profile", "user_preferences", "user_profile"):
        val = meta.get(key)
        if isinstance(val, dict):
            for k in ("display_name", "name", "username", "displayName", "userName"):
                v = val.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()[:50]
        if isinstance(val, str) and val.strip() and len(val) <= 50:
            return val.strip()[:50]
    return None


def _parse_named_selection(raw_text: str, max_idx: int) -> tuple[str, str, list[int], list[int]] | None:
    """
    Parse user input containing both name and item indices.
    Supported formats:
    - "RNF: 1 2 3"
    - "I'm RNF 1 2 3"
    - "I am RNF 1 2 3"
    - "RNF 1 2 3"
    - "RNF took 1 2 3"
    - "RNF has 1 2 3"
    """
    text = raw_text.strip()
    if not text:
        return None

    # Do not treat command-like messages as selection input.
    if re.match(
        r"^(add|edit|update|change|remove|delete|del|reset|pending|status|calculate|group\s*size|participants)\b",
        text,
        re.IGNORECASE,
    ):
        return None

    def parse_selection(selection_text: str) -> tuple[list[int], list[int]]:
        indices = parse_item_selection(selection_text, max_idx)
        raw_numbers = sorted({int(x) for x in re.findall(r"\d+", selection_text)})
        invalid = [n for n in raw_numbers if n < 1 or n > max_idx]
        return indices, invalid

    # 1) Name: numbers
    m = re.match(r"^([^:]{1,50})\s*:\s*(.+)$", text)
    if m:
        name = m.group(1).strip()
        selection_text = m.group(2).strip()
        indices, invalid = parse_selection(selection_text)
        if name and indices:
            return name, selection_text, indices, invalid

    # 2) I'm Name 1 2 3  /  I am Name 1 2 3
    m = re.match(r"^(?:i'?m|i\s+am)\s+(.+)$", text, re.IGNORECASE)
    if m:
        rest = m.group(1).strip()
        # split before trailing number sequence
        m2 = re.match(r"^(.+?)\s+((?:\d+[\s,;]*)+)$", rest)
        if m2:
            name = m2.group(1).strip()
            selection_text = m2.group(2).strip()
            indices, invalid = parse_selection(selection_text)
            if name and indices:
                return name, selection_text, indices, invalid

    # 2b) Name took/has/brought numbers
    m = re.match(r"^(.+?)\s+(?:took|has|brought|got)\s+(.+)$", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        selection_text = m.group(2).strip()
        indices, invalid = parse_selection(selection_text)
        if name and indices:
            return name, selection_text, indices, invalid

    # 3) Name 1 2 3
    m = re.match(r"^(.+?)\s+((?:\d+[\s,;]*)+)$", text)
    if m:
        name = m.group(1).strip()
        selection_text = m.group(2).strip()
        indices, invalid = parse_selection(selection_text)
        # Guard against pure numeric first token becoming a "name"
        if name and not re.fullmatch(r"[\d\s,;]+", name) and indices:
            return name, selection_text, indices, invalid

    return None


def _parse_indices_with_validation(text: str, max_idx: int) -> tuple[list[int], list[int]]:
    """Returns (valid_indices, invalid_numbers_out_of_range)."""
    indices = parse_item_selection(text, max_idx)
    raw_numbers = sorted({int(x) for x in re.findall(r"\d+", text)})
    invalid = [n for n in raw_numbers if n < 1 or n > max_idx]
    return indices, invalid


def _remap_selections_after_item_removal(
    selections: dict[str, list[int]],
    removed_idx: int,
) -> dict[str, list[int]]:
    """
    Keep selections when an item is removed:
    - removed item index is dropped
    - indexes above removed_idx shift down by 1
    """
    updated: dict[str, list[int]] = {}
    for sid, idx_list in selections.items():
        remapped: list[int] = []
        for n in idx_list:
            if n == removed_idx:
                continue
            if n > removed_idx:
                remapped.append(n - 1)
            else:
                remapped.append(n)
        # Keep deterministic unique order
        updated[sid] = sorted(set(remapped))
    return updated


def _find_item_indexes_by_name(receipt: Receipt, query: str) -> list[int]:
    """Find item indexes by name (exact first, then contains). Returns 1-based indexes."""
    q = query.strip().lower()
    if not q:
        return []
    exact = [i.index for i in receipt.items if i.name.strip().lower() == q]
    if exact:
        return exact
    return [i.index for i in receipt.items if q in i.name.strip().lower()]


def _looks_like_non_selection_command(text: str) -> bool:
    """
    Detect command-like messages that should never be interpreted as item selections.
    This prevents cases like 'calculate group size 2' from being parsed as selection [2].
    """
    t = text.strip().lower()
    if _extract_calculate_mode(t):
        return True
    if _extract_group_size_flexible(t) is not None:
        return True
    if _intent_is_pending(t) or _intent_is_start_poll(t) or _intent_is_new_receipt(t) or _intent_is_help(t):
        return True
    return bool(
        re.match(
            r"^(add|edit|update|change|remove|delete|del|reset)\b",
            t,
            re.IGNORECASE,
        )
    )


def _intent_is_help(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("help", "how to", "instructions", "guide", "what can you do")) or t in ("hi", "hello", "hey", "start")


def _intent_is_new_receipt(text: str) -> bool:
    t = text.lower().strip()
    return t in ("new receipt", "new", "start receipt", "reset") or bool(re.search(r"\b(start|create)\s+(a\s+)?new\s+receipt\b", t))


def _intent_is_start_poll(text: str) -> bool:
    t = text.lower()
    return t in ("done", "start poll", "poll", "lock") or bool(re.search(r"\b(start|begin)\s+(the\s+)?poll\b", t))


def _intent_is_pending(text: str) -> bool:
    t = text.lower()
    return t in ("pending", "status", "who is left", "who left") or bool(re.search(r"\b(who|what)\s+(is\s+)?left\b", t))


def _intent_is_show_items(text: str) -> bool:
    t = text.lower()
    return t in ("items", "list", "show items", "receipt") or bool(re.search(r"\b(show|list)\s+(the\s+)?items\b", t))


def _extract_group_size_flexible(text: str) -> int | None:
    t = text.lower()
    patterns = [
        r"(?:group\s*size|participants)\s+(\d+)",
        r"\bwe\s+are\s+(\d+)\b",
        r"\b(\d+)\s+(?:people|members|participants)\b",
    ]
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _extract_calculate_mode(text: str) -> str | None:
    """Return one of: summary, detailed, table, readable."""
    t = text.lower()
    if "calculate" not in t and "split" not in t and "summary" not in t and "detailed" not in t and "table" not in t:
        return None
    if "summary" in t:
        return "summary"
    if "readable" in t:
        return "readable"
    if "table" in t:
        return "table"
    if "detailed" in t:
        return "detailed"
    # default when user says calculate/split in free text
    if "calculate" in t or "split" in t or "result" in t:
        return "readable"
    return None


def _extract_payer_name_from_text(raw_text: str) -> str | None:
    """
    Extract payer name from flexible text, e.g.
    - "paid by rutuja"
    - "this bill is paid by rutuja please continue"
    - "rutuja paid this bill"
    - "payer is rutuja"
    """
    text = raw_text.strip()
    if not text:
        return None

    patterns = [
        r"\bpaid\s+by\s+([A-Za-z][A-Za-z0-9 _-]{0,80})",
        r"\b([A-Za-z][A-Za-z0-9 _-]{0,80})\s+paid\s+(?:this\s+)?bill\b",
        r"\bpayer\s*(?:is|:)\s*([A-Za-z][A-Za-z0-9 _-]{0,80})",
    ]

    candidate = None
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            break

    if not candidate:
        return None

    # Trim trailing command-like words if present in same message.
    candidate = re.split(
        r"\b(group\s*size|participants|done|start\s+poll|calculate|pending|please|thanks|thank\s+you|continue|proceed|then)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;!-")

    if not candidate or not re.search(r"[A-Za-z]", candidate):
        return None
    return candidate[:50]


def _pending_status_text(
    names: dict[str, str],
    participants: list[str],
    selections: dict[str, list[int]],
    expected_count: int | None = None,
) -> str:
    """Human-readable pending checklist during polling."""
    if not participants:
        return "No participants seen yet."
    pending = [sid for sid in participants if sid not in selections]
    responded = [sid for sid in participants if sid in selections]

    def label(sid: str, idx: int) -> str:
        return names.get(sid) or f"Friend {idx}"

    total_expected = expected_count if isinstance(expected_count, int) and expected_count > 0 else len(participants)

    lines = ["**Polling status**", ""]
    lines.append(f"Responded: {len(responded)} / {total_expected}")
    if total_expected > len(participants):
        lines.append(f"Seen in chat so far: {len(participants)}")

    if pending:
        lines.append("")
        lines.append("Pending:")
        for i, sid in enumerate(pending, 1):
            lines.append(f"- {label(sid, i)}")
    else:
        lines.append("")
        remaining_unknown = max(total_expected - len(responded), 0)
        if remaining_unknown > 0:
            lines.append(f"Waiting for {remaining_unknown} more participant(s) to reply.")
        else:
            lines.append("Everyone has responded. You can run **calculate summary** or **calculate detailed**.")
    return "\n".join(lines)


def _build_poll_text(receipt: Receipt) -> str:
    """Reusable poll prompt."""
    num_list = " ".join(str(i) for i in range(1, len(receipt.items) + 1))
    return (
        "📋 **Which items did you bring?**\n\n"
        f"{receipt.format_items()}\n\n"
        "Reply with name and numbers.\n"
        "Examples: **RNF: 1 2 3** or **I'm RNF 1 2 3**\n\n"
        f"Quick copy: `{num_list}` — delete the numbers you didn't bring.\n\n"
        "Next: each person sends one reply. Run **pending** to see who is left.\n"
        "Optional: set total participants using **group size 3**.\n"
        "Then run **calculate summary** or **calculate detailed**."
    )


WELCOME = """🧾 **Receipt / Expense Calculator**

Send a **photo of your receipt** and I’ll extract the line items, or add items manually.

**With photo:** Just attach the receipt image → I’ll list items → reply **done** to start the poll.

**Manual:** Say **new receipt**, then **add &lt;name&gt; &lt;price&gt;** (e.g. `add Pizza 12`).

**Poll:** Say **done**. Each person replies with name and numbers.
Examples: `RNF: 1 2 3`, `I'm RNF 1 2 3`, `RNF took 1 2 3`.
Optional: set expected participants with `group size 3`.

**Payer:** Set who paid the bill.
Examples: `Rutuja paid this bill` or `I paid this bill`.

**Split:** Use `calculate summary` for compact totals table, `calculate detailed` (or `calculate table`) for full tables, or `calculate readable` for narrative breakdown.

**Corrections:** `remove item 3`, `edit item 3 price 4.99`, `reset me`, `reset <name>`

**Commands:** `new receipt` · `add &lt;name&gt; &lt;price&gt;` · `done` · `group size 3` · `Rutuja paid this bill` · `Name: 1 2 3` · `pending` · `calculate summary` · `calculate detailed` · `calculate table` · `calculate readable` · `help`"""


@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    if sender == ctx.agent.address:
        return

    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    # Collect content: session start, text, image resource, and metadata (for display name)
    user_text = ""
    image_bytes: bytes | None = None
    display_name_from_metadata: str | None = None
    saw_attachment = False  # True if we saw any resource (even if download failed)

    # ASI-One can send user profile metadata at top-level message metadata.
    top_level_meta = getattr(msg, "metadata", None)
    if isinstance(top_level_meta, dict):
        top_level_name = _display_name_from_metadata(top_level_meta)
        if top_level_name:
            display_name_from_metadata = top_level_name

    for item in msg.content:
        if isinstance(item, MetadataContent):
            meta = item.metadata or {}
            content_level_name = _display_name_from_metadata(meta)
            if content_level_name:
                display_name_from_metadata = content_level_name
        elif isinstance(item, StartSessionContent):
            await ctx.send(
                sender,
                ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[MetadataContent(type="metadata", metadata={"attachments": "true"})],
                ),
            )
            await ctx.send(sender, text_msg(WELCOME))
            # Continue to process image if present in same message
        elif isinstance(item, TextContent):
            user_text = (item.text or "").strip()
        elif isinstance(item, ResourceContent):
            saw_attachment = True
            ctx.logger.info(f"Received resource: {item.resource_id}")
            data = download_image_resource(ctx, item)
            if data:
                image_bytes = data
        elif isinstance(item, dict) and (item.get("type") == "resource" or item.get("type") == "image"):
            # Content may arrive as raw dict from some clients (e.g. ASI-One)
            saw_attachment = True
            rid = item.get("resource_id") or item.get("resourceId")
            res = item.get("resource")
            if rid or res:
                resources_list = None
                if res is not None:
                    resources_list = res if isinstance(res, list) else [res]
                if rid:
                    ctx.logger.info(f"Received resource (dict): {rid}")
                    data = _download_image_bytes(ctx, str(rid), resources_list)
                else:
                    data = _download_image_bytes(ctx, "", resources_list) if resources_list else None
                if data:
                    image_bytes = data
            else:
                ctx.logger.warning("Resource dict missing resource_id and resource/uri")
        else:
            ctx.logger.debug(f"Content item type: {type(item).__name__}")

    # Prepare cleaned text once so image+text in same message can reuse parsing.
    raw_text = user_text.strip()
    raw_text = re.sub(r"@\w+", "", raw_text).strip()
    text = raw_text.lower()

    # Use display name from profile/metadata automatically (no need to type "I'm X")
    # Don't overwrite if user explicitly set name with "I'm X"
    if display_name_from_metadata:
        user_set = set(ctx.storage.get(RECEIPT_NAMES_USER_SET) or [])
        if sender not in user_set:
            names = ctx.storage.get(RECEIPT_NAMES) or {}
            names[sender] = display_name_from_metadata
            ctx.storage.set(RECEIPT_NAMES, names)
            ctx.logger.info(f"Display name from profile for {sender[:12]}…: {display_name_from_metadata[:20]}")

    # --- Process receipt photo if present ---
    if image_bytes:
        try:
            items_found = extract_items_from_receipt_image(image_bytes)
        except Exception as e:
            ctx.logger.exception("Vision extraction failed")
            await ctx.send(
                sender,
                text_msg(f"Could not read the receipt image. Make sure OPENAI_API_KEY is set. Error: {e}. You can add items manually with **add &lt;name&gt; &lt;price&gt;**."),
            )
            items_found = []
        if items_found:
            receipt = Receipt()
            for name, price in items_found:
                receipt.add_item(name, price)
            _save_receipt(ctx, "draft", receipt, {})
            ctx.storage.set(RECEIPT_PARTICIPANTS, [])
            ctx.storage.set(RECEIPT_EXPECTED_COUNT, None)
            ctx.storage.set(RECEIPT_PAYER_SENDER, None)
            ctx.storage.set(RECEIPT_PAYER_NAME, None)

            # If text came with the image, capture details and skip steps when possible.
            names = ctx.storage.get(RECEIPT_NAMES) or {}
            received_parts: list[str] = ["receipt image"]

            # Payer hints from first message text
            payer_set = False
            if re.match(r"^i\s+paid\s+this\s+bill\.?$", raw_text, re.IGNORECASE):
                payer_name = names.get(sender) or "Payer"
                ctx.storage.set(RECEIPT_PAYER_SENDER, sender)
                ctx.storage.set(RECEIPT_PAYER_NAME, payer_name)
                received_parts.append(f"payer = {payer_name}")
                payer_set = True
            else:
                payer_name = _extract_payer_name_from_text(raw_text)
                if payer_name:
                    payer_sender = None
                    for sid, nm in names.items():
                        if (nm or "").strip().lower() == payer_name.lower():
                            payer_sender = sid
                            break
                    ctx.storage.set(RECEIPT_PAYER_SENDER, payer_sender)
                    ctx.storage.set(RECEIPT_PAYER_NAME, payer_name)
                    received_parts.append(f"payer = {payer_name}")
                    payer_set = True

            # Group size hint from first message text
            group_set = False
            expected = _extract_group_size_flexible(text)
            if expected is not None:
                if 1 <= expected <= 50:
                    ctx.storage.set(RECEIPT_EXPECTED_COUNT, expected)
                    received_parts.append(f"group size = {expected}")
                    group_set = True

            # If user said done/start poll in same first message, skip to polling immediately
            start_poll_now = _intent_is_start_poll(text)
            if start_poll_now:
                _save_receipt(ctx, "polling", receipt, {})
                status_line = ", ".join(received_parts)
                poll_intro = f"📷 **Read your receipt.**\n\n✅ Received: {status_line}\n\n"
                await ctx.send(sender, text_msg(poll_intro + _build_poll_text(receipt)))
                return

            # Single message with item list so the list always shows (no separate "Reading..." message)
            lines = [
                "📷 **Read your receipt.**",
                "",
                "✅ **Received and processed:**",
                "",
                f"- Receipt image with {len(receipt.items)} items",
                f"- Total detected: ${receipt.total():.2f}",
            ]
            if payer_set:
                lines.append("- Payer details received")
            if group_set:
                lines.append("- Group size received")
            lines.append("")
            lines.append("**Items detected:**")
            lines.append(receipt.format_items())
            lines.append("")
            lines.append("**Next:**")
            if not payer_set:
                lines.append("- If known, set payer: `Rutuja paid this bill` (or `I paid this bill`)")
            if not group_set:
                lines.append("- Optional: set group size for pending accuracy: `group size 3`")
            lines.append("- Say **done** to start the poll, or add/edit items if needed")
            await ctx.send(sender, text_msg("\n".join(lines)))
        else:
            await ctx.send(
                sender,
                text_msg("I couldn’t find any line items in that image. Try a clearer photo or add items manually: **add &lt;name&gt; &lt;price&gt;**."),
            )
        return

    # --- Text-only handling ---
    if not raw_text:
        if saw_attachment and not image_bytes:
            await ctx.send(
                sender,
                text_msg(
                    "I received an attachment but couldn’t load the image. "
                    "Check that the file is a supported image (JPEG, PNG, etc.). "
                    "You can also add items manually: **add &lt;name&gt; &lt;price&gt;**."
                ),
            )
        else:
            await ctx.send(sender, text_msg("Send a receipt photo or type **help** for commands."))
        return

    state, receipt, selections = _load_receipt(ctx)

    if _intent_is_help(text):
        await ctx.send(sender, text_msg(WELCOME))
        return

    if _intent_is_new_receipt(text):
        _save_receipt(ctx, "draft", Receipt(), {})
        ctx.storage.set(RECEIPT_PARTICIPANTS, [])
        ctx.storage.set(RECEIPT_EXPECTED_COUNT, None)
        ctx.storage.set(RECEIPT_PAYER_SENDER, None)
        ctx.storage.set(RECEIPT_PAYER_NAME, None)
        await ctx.send(sender, text_msg("✅ New receipt started. Add items with **add &lt;name&gt; &lt;price&gt;** or send a receipt photo."))
        return

    # Optional: explicitly set total expected participants in this round
    expected = _extract_group_size_flexible(text)
    if expected is not None and state in ("draft", "polling", "done"):
        if expected < 1:
            await ctx.send(sender, text_msg("Group size must be at least 1."))
            return
        if expected > 50:
            await ctx.send(sender, text_msg("Group size is too large. Use a value up to 50."))
            return
        ctx.storage.set(RECEIPT_EXPECTED_COUNT, expected)
        await ctx.send(
            sender,
            text_msg(
                f"✅ Expected participants set to **{expected}** for this receipt.\n"
                "Use **pending** to check progress."
            ),
        )
        return

    # Set payer command
    # Examples: "Rutuja paid this bill", "I paid this bill"
    if state in ("draft", "polling", "done"):
        names = ctx.storage.get(RECEIPT_NAMES) or {}
        if re.match(r"^i\s+paid\s+this\s+bill\.?$", raw_text, re.IGNORECASE):
            payer_name = names.get(sender) or "Payer"
            ctx.storage.set(RECEIPT_PAYER_SENDER, sender)
            ctx.storage.set(RECEIPT_PAYER_NAME, payer_name)
            await ctx.send(sender, text_msg(f"✅ Payer set to **{payer_name}**."))
            return

        payer_name = _extract_payer_name_from_text(raw_text)
        if payer_name:
            payer_sender = None
            for sid, nm in names.items():
                if (nm or "").strip().lower() == payer_name.lower():
                    payer_sender = sid
                    break
            # If this sender is naming themselves, map to sender as payer
            if not payer_sender and names.get(sender) and names.get(sender).strip().lower() == payer_name.lower():
                payer_sender = sender
            ctx.storage.set(RECEIPT_PAYER_SENDER, payer_sender)
            ctx.storage.set(RECEIPT_PAYER_NAME, payer_name)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Payer set to **{payer_name}**.\n"
                    "Tip: if payer is in this group, they should also submit their items so settlement is exact."
                ),
            )
            return

    add_match = re.match(r"add\s+(.+?)\s+\$?([\d.]+)\s*$", text, re.I)
    if add_match:
        name_part, price_str = add_match.group(1).strip(), add_match.group(2)
        name_part = re.sub(r"^\$", "", name_part).strip()
        try:
            price = Decimal(price_str)
            if price <= 0:
                await ctx.send(sender, text_msg("Price must be positive."))
                return
            receipt.add_item(name_part, price)
            if state == "draft":
                _save_receipt(ctx, state, receipt, selections)
                await ctx.send(
                    sender,
                    text_msg(
                        f"✅ Added: **{receipt.items[-1].name}** — ${receipt.items[-1].price:.2f}\n\n"
                        f"{receipt.format_items()}\n\n"
                        "Next: add more items or say **done** to start polling."
                    ),
                )
            else:
                _save_receipt(ctx, state, receipt, selections)
                await ctx.send(
                    sender,
                    text_msg(
                        f"✅ Added: **{receipt.items[-1].name}** — ${receipt.items[-1].price:.2f}\n\n"
                        f"{receipt.format_items()}\n\n"
                        "Existing selections were kept."
                    ),
                )
        except Exception as e:
            await ctx.send(sender, text_msg(f"Use: **add ItemName 12.50** (e.g. add Pizza 12). Error: {e}"))
        return

    # Edit/remove item commands
    remove_match = re.match(r"(?:remove|delete|del)\s+(?:item\s+)?(\d+)\s*$", text, re.I)
    if remove_match:
        idx = int(remove_match.group(1))
        if idx < 1 or idx > len(receipt.items):
            await ctx.send(sender, text_msg(f"Item {idx} does not exist. Valid range is 1-{len(receipt.items)}."))
            return
        removed = receipt.items.pop(idx - 1)
        for i, it in enumerate(receipt.items, 1):
            it.index = i
        if state == "draft":
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Removed item {idx}: **{removed.name}**\n\n{receipt.format_items()}\n\n"
                    "Next: add/edit/remove items or say **done**."
                ),
            )
        else:
            selections = _remap_selections_after_item_removal(selections, idx)
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Removed item {idx}: **{removed.name}**\n\n{receipt.format_items()}\n\n"
                    "Existing selections were updated and kept."
                ),
            )
        return

    edit_match = re.match(
        r"(?:edit|update|change)\s+item\s+(\d+)\s+(?:price\s+)?(?:to\s+)?\$?([\d.]+)\s*$",
        text,
        re.I,
    )
    if edit_match:
        idx = int(edit_match.group(1))
        new_price = Decimal(edit_match.group(2))
        if idx < 1 or idx > len(receipt.items):
            await ctx.send(sender, text_msg(f"Item {idx} does not exist. Valid range is 1-{len(receipt.items)}."))
            return
        if new_price <= 0:
            await ctx.send(sender, text_msg("Price must be positive."))
            return
        receipt.items[idx - 1].price = new_price
        if state == "draft":
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Updated item {idx} price to ${new_price:.2f}\n\n{receipt.format_items()}\n\n"
                    "Next: add/edit/remove items or say **done**."
                ),
            )
        else:
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Updated item {idx} price to ${new_price:.2f}\n\n{receipt.format_items()}\n\n"
                    "Existing selections were kept."
                ),
            )
        return

    # Name-based price update for OCR mistakes:
    # "update price of cauliflower florets to 2.50"
    # "set price for cauliflower florets to $2.50"
    # "change price of avocado bag to 4.99"
    edit_name_match = re.match(
        r"(?:update|change|set|edit)\s+price\s+(?:of|for)\s+(.+?)\s+(?:to\s+)?\$?([\d.]+)\s*$",
        text,
        re.I,
    )
    if edit_name_match:
        query = edit_name_match.group(1).strip()
        new_price = Decimal(edit_name_match.group(2))
        if new_price <= 0:
            await ctx.send(sender, text_msg("Price must be positive."))
            return

        matches = _find_item_indexes_by_name(receipt, query)
        if not matches:
            await ctx.send(
                sender,
                text_msg(
                    f"No item matched **{query}**. Use `items` to view names, or use index-based update like `edit item 3 to $2.50`."
                ),
            )
            return
        if len(matches) > 1:
            opts = ", ".join(str(i) for i in matches[:6])
            await ctx.send(
                sender,
                text_msg(
                    f"Multiple items matched **{query}** (items: {opts}). Use exact index, e.g. `edit item {matches[0]} to $2.50`."
                ),
            )
            return

        idx = matches[0]
        receipt.items[idx - 1].price = new_price
        if state == "draft":
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Updated item {idx} (**{receipt.items[idx - 1].name}**) price to ${new_price:.2f}\n\n{receipt.format_items()}\n\n"
                    "Next: add/edit/remove items or say **done**."
                ),
            )
        else:
            _save_receipt(ctx, state, receipt, selections)
            await ctx.send(
                sender,
                text_msg(
                    f"✅ Updated item {idx} (**{receipt.items[idx - 1].name}**) price to ${new_price:.2f}\n\n{receipt.format_items()}\n\n"
                    "Existing selections were kept."
                ),
            )
        return

    if _intent_is_start_poll(text):
        if state != "draft":
            await ctx.send(sender, text_msg("Poll already started. Reply with your item numbers (e.g. 1,2,3) or **calculate** to see the split."))
            return
        if not receipt.items:
            await ctx.send(sender, text_msg("Add at least one item first, or send a receipt photo."))
            return
        _save_receipt(ctx, "polling", receipt, selections)
        # Start with empty participant list; add people when they actually submit item selections.
        ctx.storage.set(RECEIPT_PARTICIPANTS, [])
        await ctx.send(sender, text_msg(_build_poll_text(receipt)))
        return

    if state in ("polling", "done") and receipt.items:
        max_idx = max(i.index for i in receipt.items)
        names = ctx.storage.get(RECEIPT_NAMES) or {}
        participants = ctx.storage.get(RECEIPT_PARTICIPANTS) or []
        expected_count = ctx.storage.get(RECEIPT_EXPECTED_COUNT)

        # Polling status command
        if _intent_is_pending(text):
            await ctx.send(sender, text_msg(_pending_status_text(names, participants, selections, expected_count)))
            return

        # Do not parse command-like messages as selections.
        # Let them continue to dedicated command handlers later.
        if _looks_like_non_selection_command(text):
            pass
        else:
            named_selection = _parse_named_selection(raw_text, max_idx)
            if named_selection:
                provided_name, selection_text, indices, invalid = named_selection
                if invalid:
                    await ctx.send(
                        sender,
                        text_msg(
                            f"Invalid item numbers: {', '.join(str(n) for n in invalid)}. "
                            f"Valid range is 1-{max_idx}."
                        ),
                    )
                    return
                names[sender] = provided_name
                ctx.storage.set(RECEIPT_NAMES, names)
                user_set = set(ctx.storage.get(RECEIPT_NAMES_USER_SET) or [])
                user_set.add(sender)
                ctx.storage.set(RECEIPT_NAMES_USER_SET, list(user_set))

                selections[sender] = indices
                _save_receipt(ctx, state, receipt, selections)
                participants = list(dict.fromkeys(participants + [sender]))
                ctx.storage.set(RECEIPT_PARTICIPANTS, participants)
                await ctx.send(
                    sender,
                    text_msg(
                        f"✅ Recorded your items: **{', '.join(str(i) for i in indices)}** for **{provided_name}**. "
                        f"\n\n{_pending_status_text(names, participants, selections, expected_count)}"
                    ),
                )
                return

            # If user sends only numbers, require name first so final output has names only
            indices, invalid = _parse_indices_with_validation(raw_text, max_idx)
            if invalid:
                await ctx.send(
                    sender,
                    text_msg(
                        f"Invalid item numbers: {', '.join(str(n) for n in invalid)}. "
                        f"Valid range is 1-{max_idx}."
                    ),
                )
                return
            if indices and not names.get(sender):
                await ctx.send(
                    sender,
                    text_msg(
                        "Please send your response with name and item numbers.\n"
                        "Examples: **RNF: 1 2 3** or **I'm RNF 1 2 3**"
                    ),
                )
                return
            if indices and names.get(sender):
                selections[sender] = indices
                _save_receipt(ctx, state, receipt, selections)
                participants = list(dict.fromkeys(participants + [sender]))
                ctx.storage.set(RECEIPT_PARTICIPANTS, participants)
                display = names.get(sender)
                await ctx.send(
                    sender,
                    text_msg(
                        f"✅ Recorded your items: **{', '.join(str(i) for i in indices)}** for **{display}**. "
                        f"\n\n{_pending_status_text(names, participants, selections, expected_count)}"
                    ),
                )
                return

        # Reset selection commands
        if text in ("reset me", "reset my selection"):
            if sender in selections:
                selections.pop(sender, None)
                _save_receipt(ctx, state, receipt, selections)
                await ctx.send(
                    sender,
                    text_msg(
                        "✅ Your selection was reset. Send your items again using name + numbers, "
                        "then run **calculate summary** or **calculate detailed**."
                    ),
                )
            else:
                await ctx.send(sender, text_msg("You do not have a saved selection yet."))
            return
        reset_name_match = re.match(r"reset\s+(.+)$", raw_text, re.I)
        if reset_name_match and text not in ("reset me", "reset my selection"):
            target_name = reset_name_match.group(1).strip().lower()
            target_sender = None
            for sid, nm in names.items():
                if (nm or "").strip().lower() == target_name:
                    target_sender = sid
                    break
            if target_sender and target_sender in selections:
                selections.pop(target_sender, None)
                _save_receipt(ctx, state, receipt, selections)
                await ctx.send(
                    sender,
                    text_msg(
                        f"✅ Reset selection for **{names.get(target_sender)}**. "
                        "Ask them to send updated items, then run calculate again."
                    ),
                )
            else:
                await ctx.send(sender, text_msg("Could not find that name in saved selections."))
            return

    calc_mode = _extract_calculate_mode(text)
    if calc_mode:
        if state not in ("polling", "done"):
            await ctx.send(sender, text_msg("Add items (or send a receipt photo), say **done**, then have everyone reply with their item numbers. Then say **calculate**."))
            return
        if not receipt.items:
            await ctx.send(sender, text_msg("No items on the receipt."))
            return
        if not selections:
            await ctx.send(sender, text_msg("No one has replied with their items yet. Ask everyone to reply with numbers (e.g. 1,2,3)."))
            return
        expected_count = ctx.storage.get(RECEIPT_EXPECTED_COUNT)
        if isinstance(expected_count, int) and expected_count > len(selections):
            remaining = expected_count - len(selections)
            await ctx.send(
                sender,
                text_msg(
                    f"Still waiting for **{remaining}** participant(s). "
                    "Use **pending** to see current status."
                ),
            )
            return
        names = ctx.storage.get(RECEIPT_NAMES) or {}
        missing_name_senders = [sid for sid in selections.keys() if not names.get(sid)]
        if missing_name_senders:
            await ctx.send(
                sender,
                text_msg(
                    "Some participants are missing names. Ask them to resend with name and item numbers.\n"
                    "Examples: **RNF: 1 2 3** or **I'm RNF 1 2 3**"
                ),
            )
            return
        resolved: dict[str, str] = {}
        for sid in selections.keys():
            resolved[sid] = names.get(sid)
        payer_sender_id = ctx.storage.get(RECEIPT_PAYER_SENDER)
        payer_display_name = ctx.storage.get(RECEIPT_PAYER_NAME)
        participants_for_calc = list(selections.keys())
        if calc_mode == "summary":
            result = format_split_summary_table(
                receipt,
                selections,
                resolved,
                all_participants=participants_for_calc,
                payer_sender_id=payer_sender_id,
                payer_display_name=payer_display_name,
            )
        elif calc_mode in ("detailed", "table"):
            result = format_split_full_table(
                receipt,
                selections,
                resolved,
                all_participants=participants_for_calc,
                payer_sender_id=payer_sender_id,
                payer_display_name=payer_display_name,
            )
        elif calc_mode == "readable":
            result = format_split_result(
                receipt,
                selections,
                resolved,
                all_participants=participants_for_calc,
                payer_sender_id=payer_sender_id,
                payer_display_name=payer_display_name,
            )
        else:
            # Default calculate keeps readable output for backward compatibility.
            result = format_split_result(
                receipt,
                selections,
                resolved,
                all_participants=participants_for_calc,
                payer_sender_id=payer_sender_id,
                payer_display_name=payer_display_name,
            )
        _save_receipt(ctx, "done", receipt, selections)
        await ctx.send(sender, text_msg(result))
        return

    if _intent_is_show_items(text):
        if not receipt.items:
            await ctx.send(sender, text_msg("No receipt yet. Send a receipt photo or say **new receipt** then **add &lt;name&gt; &lt;price&gt;**."))
            return
        msg_text = f"**Current receipt ({state}):**\n\n{receipt.format_items()}\n\nTotal: ${receipt.total():.2f}"
        if state == "polling" and selections:
            msg_text += f"\n\nReplied: {len(selections)} person(s). Say **calculate** to see the split."
        await ctx.send(sender, text_msg(msg_text))
        return

    # Use raw_text so name keeps original case. Match "I'm X" or "I am X" (overrides profile name)
    name_match = re.match(r"(?:i'?m|i\s+am)\s+(.+)$", raw_text, re.IGNORECASE)
    if name_match and state in ("draft", "polling", "done"):
        names = ctx.storage.get(RECEIPT_NAMES) or {}
        names[sender] = name_match.group(1).strip()[:50]
        ctx.storage.set(RECEIPT_NAMES, names)
        user_set = set(ctx.storage.get(RECEIPT_NAMES_USER_SET) or [])
        user_set.add(sender)
        ctx.storage.set(RECEIPT_NAMES_USER_SET, list(user_set))
        await ctx.send(sender, text_msg(f"✅ I'll show you as **{names[sender]}** in the split."))
        return

    if state == "polling":
        await ctx.send(
            sender,
            text_msg(
                "Reply with name and item numbers. Examples: **RNF: 1 2 3** or **I'm RNF 1 2 3**.\n"
                "Use **pending** to see who is left."
            ),
        )
        return

    await ctx.send(sender, text_msg("Send a receipt photo or type **help** for commands."))


@chat_proto.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"ACK from {sender}")


agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__":
    print("🧾 Receipt / Expense Calculator")
    print("Runnable on Agentverse + ASI-One. Send a receipt photo or add items manually.")
    print("https://asi1.ai  |  https://agentverse.ai")
    print("\nPress Ctrl+C to stop\n")
    agent.run()
