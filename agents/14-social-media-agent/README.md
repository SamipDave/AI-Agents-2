# Social Media Content Agent

Generates platform-optimized social media content from any topic. An optional
Xquik research step searches recent X posts before CrewAI builds the strategy.

**Framework**: CrewAI  
**LLM**: GPT-4o-mini  

## Setup

Use Python 3.10-3.12. CrewAI 0.80.0 does not install on Python 3.13 patch
releases.

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Set `XQUIK_API_KEY` only when using
`--x-query`. Create an Xquik API key in the
[Xquik dashboard](https://dashboard.xquik.com/en/account?tab=api-keys).

## Run

```bash
python agent.py --topic "The future of remote work"
python agent.py --topic "Product launch announcement" --brand "YourBrand" --platforms "twitter,linkedin"
python agent.py --topic "Open source AI agents" --x-query '"open source" AI agents' --x-limit 10
```

`--x-query` sends only that query to Xquik. It retrieves 1-25 posts in latest
order through the published X search API. The API key stays in the request
header and never enters the LLM prompt. The client rejects redirects before
processing a response. The returned post text, author, metrics, and source URL
do enter the OpenAI prompt as research context.

The agent treats returned posts as untrusted research, not instructions. It
escapes prompt delimiters, caps each post at 600 characters, and asks the
strategist to preserve source URLs for human review. Review claims and output
before publishing.

## Offline smoke test

The smoke test makes no network or LLM calls:

```bash
python agent.py --self-check
```

Expected output:

```text
self-check OK
```

It verifies the Xquik request parameters, 25-post cap, API-key isolation,
source URL construction, prompt-delimiter escaping, and platform selection.

## Safety notes

Treat this agent as a drafting tool. Review every claim, source, and post before
publishing. Search results can include inaccurate, biased, offensive, or
personal content. Use only data you may lawfully process. Do not use the agent
for automated harassment, impersonation, surveillance, or high-impact
decisions.
