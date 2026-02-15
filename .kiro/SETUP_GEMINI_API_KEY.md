# Setting Up Gemini API Key

## Error You're Seeing

```
Fathom Error: No Gemini API key provided and no Vertex AI credentials found.
Please set GEMINI_API_KEY environment variable or configure Google Cloud credentials.
```

## Solution: Set GEMINI_API_KEY Environment Variable

### Option 1: Create .env File (Recommended)

Create a `.env` file in the project root directory:

```bash
# In the project root (where pyproject.toml is)
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual Gemini API key.

### Option 2: Export Environment Variable

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Add this to your `~/.zshrc` or `~/.bashrc` to make it permanent.

### Option 3: Pass via Command Line

```bash
fathom run "your intent" --api-key "your_api_key_here"
```

## How to Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key" or "Create API Key"
4. Copy the generated API key
5. Use it in one of the options above

## Verify Setup

After setting the API key, verify it works:

```bash
# Check if environment variable is set
echo $GEMINI_API_KEY

# Try running Fathom
fathom run "test intent" --serial emulator-5554
```

## Alternative: Use Vertex AI (Advanced)

If you prefer to use Google Cloud Vertex AI instead of the API key:

1. Set up Google Cloud project
2. Enable Vertex AI API
3. Create service account credentials
4. Set environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   export VERTEX_PROJECT_ID="your-project-id"
   ```

## Troubleshooting

### API Key Not Being Read

If you created a `.env` file but it's not being read:

1. Make sure the `.env` file is in the project root (same directory as `pyproject.toml`)
2. Check the file has the correct format: `GEMINI_API_KEY=your_key` (no quotes, no spaces around =)
3. Restart your terminal or reload the environment

### Invalid API Key Error

If you get an "invalid API key" error:

1. Verify the API key is correct (copy-paste from Google AI Studio)
2. Make sure there are no extra spaces or newlines
3. Check if the API key has been revoked or expired

### Rate Limit Errors

If you hit rate limits:

1. Wait a few minutes and try again
2. Consider upgrading your API quota in Google AI Studio
3. Use Vertex AI for higher quotas (requires Google Cloud setup)

## Current Configuration

The system will use these settings in order of priority:

1. Command line `--api-key` flag
2. `GEMINI_API_KEY` environment variable
3. `.env` file in project root
4. Vertex AI credentials (if configured)

If none are found, you'll get the error message.
