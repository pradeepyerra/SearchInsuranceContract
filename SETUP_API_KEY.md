# Setting Up API Keys

## Quick Setup

1. **Edit the `.env` file** in the project root:
   ```bash
   cd /Users/pradeepkumaryerra/projects/agents-main
   nano .env
   # or use your preferred editor
   ```

2. **Replace `your_openai_api_key_here`** with your actual OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Save the file** and restart the web application.

## Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)
5. Paste it in your `.env` file

## File Location

The `.env` file should be located at:
```
/Users/pradeepkumaryerra/projects/agents-main/.env
```

## After Setting Up

1. Restart the web application:
   ```bash
   cd 4_langgraph
   python3 agent_web_app.py
   ```

2. The application will automatically load the API key from the `.env` file.

## Optional: Serper API Key (for Internet Search)

If you want to use internet search features:

1. Get a free API key from https://serper.dev
2. Add it to your `.env` file:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   ```

## Security Note

⚠️ **Never commit your `.env` file to version control!** It contains sensitive API keys.

The `.env` file is already in `.gitignore` to prevent accidental commits.

