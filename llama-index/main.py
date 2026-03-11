from config import agent, chat_proto, payment_proto, QDRANT_URL, FET_USE_TESTNET, ANALYSIS_FEE
from payment import set_agent_wallet
from handlers import *  # noqa: F401,F403 — registers protocol handlers

from uagents import Context

agent.include(chat_proto, publish_manifest=True)
agent.include(payment_proto, publish_manifest=True)
set_agent_wallet(agent.wallet)


@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Fetch RAG Agent started — address: {agent.address}")
    ctx.logger.info(f"Wallet: {agent.wallet.address() if agent.wallet else 'N/A'}")
    ctx.logger.info(f"Network: {'testnet' if FET_USE_TESTNET else 'mainnet'}")
    ctx.logger.info(f"Qdrant: {QDRANT_URL}")
    ctx.logger.info(f"Fee: {ANALYSIS_FEE} FET per document")


if __name__ == "__main__":
    agent.run()
