# Receipt / Expense Calculator Agent

Agent for **ASI-One** and **Agentverse** that splits receipts fairly: send a **photo of a receipt** (or add items manually), then have each person mark which items they brought and see the split.

## Run locally

```bash
cd expense-calculator-group
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, AGENT_SEED, AGENT_MAILBOX_KEY
python agent.py
```

## Deploy on Agentverse + ASI-One

1. **Agentverse:** [agentverse.ai](https://agentverse.ai) → Agents → Launch an Agent → **Chat Protocol**.
2. Use **mailbox** (no public URL needed): run the agent (e.g. on your machine or a server) and connect via the Agent Inspector link from the logs.
3. **ASI-One:** After the agent is registered on Agentverse, it appears in [asi1.ai](https://asi1.ai); you can chat there or add it to a group.

See [deploy-agent-on-av/docs.md](../innovation-lab-examples/deploy-agent-on-av/docs.md) for Render/mailbox deployment and env vars.

## What the agent does

- **Receipt photo:** Attach an image → agent extracts line items (name + price) with OpenAI Vision and lists them. Reply **done** to start the poll.
- **Manual:** Say **new receipt**, then **add Pizza 12** (and more). Say **done** when finished.
- **Poll:** Each person replies with the **numbers** of items they brought (e.g. `1,2,3`). Multiple people can claim the same item.
- **Split:** Say **calculate** → agent shows each person’s share (only people who brought an item pay for it).

## Commands

| Command | Description |
|--------|-------------|
| *(photo)* | Extract items from receipt image |
| `new receipt` | Start a new receipt |
| `add <name> <price>` | Add item (e.g. `add Coffee 3.50`) |
| `done` | Lock items and start the poll |
| `1,2,3` | Your items (reply with numbers) |
| `calculate` | Show split per person |
| `I'm Alice` | Set display name |
| `help` | Show instructions |

## Env vars

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for photos) | OpenAI API key for receipt image extraction (Vision) |
| `AGENT_SEED` | Yes | Seed phrase for agent identity |
| `AGENT_MAILBOX_KEY` | Yes (for Agentverse) | Mailbox key from Agentverse |
| `AGENTVERSE_URL` | No | Default `https://agentverse.ai` |

## Project layout

```
expense-calculator-group/
├── agent.py           # uAgents chat protocol (photo + text commands)
├── expense_logic.py   # Receipt, split calculation
├── receipt_vision.py   # OpenAI Vision receipt extraction
├── requirements.txt
├── .env.example
└── README.md
```
