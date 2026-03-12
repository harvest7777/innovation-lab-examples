from decimal import Decimal
from uuid import uuid4

from uagents import Context
from uagents_core.contrib.protocols.payment import RequestPayment
from cosmpy.aerial.client import LedgerClient, NetworkConfig

from config import FET_USE_TESTNET, ANALYSIS_FEE, ACCEPTED_FUNDS

_agent_wallet = None

# Track payment sessions: reference -> session info
_payment_sessions: dict[str, dict] = {}
# Track consumed transaction IDs to prevent replay
_consumed_transactions: set[str] = set()


def set_agent_wallet(wallet):
    global _agent_wallet
    _agent_wallet = wallet


def get_agent_wallet():
    return _agent_wallet


def verify_fet_payment_to_agent(
    transaction_id: str,
    expected_amount_fet: str,
    sender_fet_address: str,
    recipient_agent_wallet,
    logger,
) -> bool:
    """Verify an on-chain FET transfer by querying the Fetch.ai ledger."""
    # Prevent replay of already-consumed transactions
    if transaction_id in _consumed_transactions:
        logger.warning(f"Transaction {transaction_id} already consumed — replay rejected")
        return False

    network_config = (
        NetworkConfig.fetchai_stable_testnet()
        if FET_USE_TESTNET
        else NetworkConfig.fetchai_mainnet()
    )
    ledger = LedgerClient(network_config)
    expected_amount_micro = int(Decimal(expected_amount_fet) * Decimal(10**18))

    try:
        tx_response = ledger.query_tx(transaction_id)
    except Exception as e:
        logger.error(f"Failed to query transaction {transaction_id}: {e}")
        return False

    if not tx_response.is_successful():
        logger.error(f"Transaction {transaction_id} was not successful")
        return False

    recipient_found = False
    amount_found = False
    sender_found = False
    denom = "atestfet" if FET_USE_TESTNET else "afet"
    expected_recipient = str(recipient_agent_wallet.address())

    for event_type, event_attrs in tx_response.events.items():
        if event_type == "transfer":
            if event_attrs.get("recipient") == expected_recipient:
                recipient_found = True
                if event_attrs.get("sender") == sender_fet_address:
                    sender_found = True
                amount_str = event_attrs.get("amount", "")
                if amount_str and amount_str.endswith(denom):
                    try:
                        amount_value = int(amount_str.replace(denom, ""))
                        if amount_value >= expected_amount_micro:
                            amount_found = True
                    except Exception:
                        pass

    verified = recipient_found and amount_found and sender_found

    if verified:
        _consumed_transactions.add(transaction_id)

    logger.info(
        f"Payment verification: recipient={recipient_found}, "
        f"amount={amount_found}, sender={sender_found} -> {'PASS' if verified else 'FAIL'}"
    )
    return verified


def create_payment_session(user_address: str) -> str:
    """Create and persist a payment session. Returns the reference ID."""
    reference = str(uuid4())
    wallet = get_agent_wallet()
    _payment_sessions[reference] = {
        "user_address": user_address,
        "amount": ANALYSIS_FEE,
        "recipient": str(wallet.address()) if wallet else "unknown",
        "consumed": False,
    }
    return reference


def validate_and_consume_session(reference: str, transaction_id: str) -> bool:
    """Validate a payment session exists and mark it consumed. Returns True if valid."""
    session = _payment_sessions.get(reference)
    if not session:
        return False
    if session["consumed"]:
        return False
    session["consumed"] = True
    session["transaction_id"] = transaction_id
    return True


async def request_payment_from_user(ctx: Context, user_address: str, description: str | None = None):
    """Send a RequestPayment message to the user via Agentverse."""
    wallet = get_agent_wallet()
    fet_network = "stable-testnet" if FET_USE_TESTNET else "mainnet"

    reference = create_payment_session(user_address)

    metadata = {"fet_network": fet_network}
    if wallet:
        metadata["provider_agent_wallet"] = str(wallet.address())

    payment_request = RequestPayment(
        accepted_funds=ACCEPTED_FUNDS,
        recipient=str(wallet.address()) if wallet else "unknown",
        deadline_seconds=300,
        reference=reference,
        description=description or f"RAG document ingestion fee: {ANALYSIS_FEE} FET",
        metadata=metadata,
    )

    # Store reference so handler can look it up
    ctx.storage.set(f"payment_ref:{user_address}", reference)

    await ctx.send(user_address, payment_request)
    ctx.logger.info(f"Payment request sent to {user_address[:20]}... ({ANALYSIS_FEE} FET, ref={reference[:8]}...)")
