import asyncio
import traceback
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Context
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
)
from uagents_core.contrib.protocols.payment import (
    CommitPayment,
    CompletePayment,
    CancelPayment,
    RejectPayment,
)

from config import chat_proto, payment_proto, ANALYSIS_FEE
from rag import (
    URL_PATTERN,
    _collection_name,
    download_document,
    ingest_document,
    query_agent,
    format_response_with_citations,
    reset_user_session,
    collection_has_points,
)
from payment import (
    verify_fet_payment_to_agent,
    request_payment_from_user,
    get_agent_wallet,
)


# ── Reply helper ──────────────────────────────────────────────────────

async def reply(ctx: Context, sender: str, text: str, *, end: bool = False):
    content = [TextContent(type="text", text=text)]
    if end:
        content.append(EndSessionContent(type="end-session"))
    await ctx.send(
        sender,
        ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=content),
    )


# ── Ingestion flow (post-payment) ────────────────────────────────────

async def run_pending_ingestion(ctx: Context, sender: str):
    pending_text = ctx.storage.get(f"pending_url:{sender}")
    if not pending_text:
        await reply(ctx, sender, "Payment received, but no pending document URL found.")
        return

    ctx.storage.remove(f"pending_url:{sender}")

    match = URL_PATTERN.search(pending_text)
    if not match:
        await reply(ctx, sender, "Could not find a valid URL in your message. Please resend.")
        return

    url = match.group(0)
    collection = _collection_name(sender)

    try:
        await reply(ctx, sender, f"Payment verified! Downloading and ingesting your document...\n\nURL: {url}")

        file_path = await asyncio.to_thread(download_document, url)
        num_chunks = await asyncio.to_thread(ingest_document, file_path, collection)

        reset_user_session(sender)

        await reply(
            ctx, sender,
            f"Document ingested successfully!\n\n"
            f"Chunks stored: {num_chunks}\n"
            f"Collection: {collection}\n\n"
            f"You can now ask me questions about the document. Try:\n"
            f'- "What are the key financial metrics?"\n'
            f'- "Summarize the risk factors"\n'
            f'- "What were the revenue figures?"',
        )
    except Exception as e:
        ctx.logger.error(f"Ingestion failed for {sender[:20]}...: {e}")
        traceback.print_exc()
        await reply(ctx, sender, f"Document ingestion failed: {e}\n\nPlease try again with a different URL.")


# ── Payment protocol handlers ────────────────────────────────────────

@payment_proto.on_message(CommitPayment)
async def handle_commit_payment(ctx: Context, sender: str, msg: CommitPayment):
    ctx.logger.info(f"CommitPayment from {sender[:20]}... tx={msg.transaction_id}")

    payment_verified = False

    if msg.funds.payment_method == "fet_direct" and msg.funds.currency == "FET":
        buyer_wallet = None
        if isinstance(msg.metadata, dict):
            buyer_wallet = msg.metadata.get("buyer_fet_wallet") or msg.metadata.get("buyer_fet_address")

        if buyer_wallet:
            payment_verified = verify_fet_payment_to_agent(
                transaction_id=msg.transaction_id,
                expected_amount_fet=ANALYSIS_FEE,
                sender_fet_address=buyer_wallet,
                recipient_agent_wallet=get_agent_wallet(),
                logger=ctx.logger,
            )
        else:
            ctx.logger.warning("No buyer wallet address in CommitPayment metadata")

    if payment_verified:
        ctx.logger.info(f"Payment VERIFIED for {sender[:20]}...")
        await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))
        await run_pending_ingestion(ctx, sender)
    else:
        ctx.logger.warning(f"Payment REJECTED for {sender[:20]}...")
        await ctx.send(
            sender,
            CancelPayment(
                transaction_id=msg.transaction_id,
                reason="Payment verification failed. Please ensure you sent the correct amount.",
            ),
        )
        await reply(ctx, sender, "Payment verification failed. Please try again.")


@payment_proto.on_message(RejectPayment)
async def handle_reject_payment(ctx: Context, sender: str, msg: RejectPayment):
    ctx.logger.info(f"Payment rejected by {sender[:20]}...")
    ctx.storage.remove(f"pending_url:{sender}")
    await reply(ctx, sender, "Payment declined. Send another document URL whenever you're ready.")


# ── Chat protocol handlers ───────────────────────────────────────────

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )

    text = "".join(item.text for item in msg.content if isinstance(item, TextContent)).strip()
    if not text:
        return

    ctx.logger.info(f"Message from {sender[:20]}...: {text[:120]}")

    # URL -> payment gate -> ingestion
    if URL_PATTERN.search(text):
        ctx.storage.set(f"pending_url:{sender}", text)
        await request_payment_from_user(
            ctx, sender,
            description=f"RAG document ingestion fee: {ANALYSIS_FEE} FET",
        )
        return

    # No URL -> query existing collection via ReAct agent
    collection = _collection_name(sender)
    if not collection_has_points(collection):
        await reply(
            ctx, sender,
            "No documents ingested yet. Send a document URL first "
            "(e.g., a PDF link) and I'll ingest it after payment.",
        )
        return

    try:
        answer, citations = await query_agent(sender, text)
        formatted = format_response_with_citations(answer, citations)
        await reply(ctx, sender, formatted)
    except Exception as e:
        ctx.logger.error(f"Query failed for {sender[:20]}...: {e}")
        traceback.print_exc()
        await reply(ctx, sender, f"An error occurred while querying: {e}")


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass
