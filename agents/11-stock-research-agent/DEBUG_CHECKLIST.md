# Debug Checklist for Stock Research Agent (Issue #139)

1. **Environment Variables**:
    - Ensure `.env` file exists in `agents/11-stock-research-agent/`.
    - Check if `OPENAI_API_KEY` is set and valid in `.env`.
    - You can verify this by running `echo $OPENAI_API_KEY` (or `printenv OPENAI_API_KEY`) after loading the env.

2. **Dependencies**:
    - Ensure you have installed requirements: `pip install -r requirements.txt`.
    - Check if there are any version conflicts.

3. **Execution**:
    - Use the provided wrapper to run `python agent.py --ticker <TICKER>`.
    - Check the console for detailed stack traces if an error occurs.
