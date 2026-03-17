# Groq-Chat-App
Groq Chat App built using Groq API and Streamlit.

## Deploy On Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and click `New app`.
3. Select:
	- Repository: your GitHub repo
	- Branch: `main`
	- Main file path: `Med/myenv/lawchatbot.py`
4. In `Advanced settings` -> `Secrets`, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

5. Click `Deploy`.

Notes:
- Dependencies are read from root `requirements.txt`.
- Python runtime is pinned in `runtime.txt`.
- Do not commit `.env` with real API keys.
