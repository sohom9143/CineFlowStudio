# CineFlow AI Studio - Agent Guidelines & Rules

## 1. Google Colab & MCP Sync Rule (Active)
- Whenever any changes, fixes, or updates are made to Jupyter Notebooks (e.g., \CineFlow_Colab_FreeTier.ipynb\ or related script modules):
  - Ensure the changes are synchronized with the Google Colab session using the \colab-mcp\ MCP server.
  - Keep the local notebook file updated and push the changes to GitHub repository (\https://github.com/sohom9143/CineFlowStudio.git\) so that cloud and local environments stay in exact sync.

## 2. Security & Credentials
- Never track or push \.env\ or secret tokens/API keys.
- Preserve \.gitignore\ protections for large binary weights, output media, and secrets.
