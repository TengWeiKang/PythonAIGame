# Security Setup Guide

This guide explains how to set up secure environment variables for the Python Webcam Detection Application.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file and add your API key:**
   ```bash
   # Required: Google Gemini API Key
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Start the application** - it will automatically use the secure environment variables.

## Security Features

### ✅ What's Secure Now

- **API keys are never stored in config files** when using environment variables
- **Input validation and sanitization** for all user prompts and file paths
- **Comprehensive error handling** prevents information disclosure
- **Resource leak prevention** with proper cleanup and retry logic
- **Content filtering** detects and filters sensitive information

### 🔒 Environment Variable Security

- The `.env` file is automatically ignored by git (add `.env` to your `.gitignore`)
- API keys from environment variables override any config file values
- Configuration files will show empty API keys when environment variables are used

## Configuration Priority

1. **Environment Variables** (.env file) - **Highest Priority**
2. Configuration File (config.json)
3. Default Values - Lowest Priority

## Environment Variables Reference

### Required
- `GEMINI_API_KEY` - Your Google Gemini API key

### Optional API Configuration
- `GEMINI_MODEL` - Model to use (default: gemini-1.5-flash)
- `GEMINI_TIMEOUT` - Request timeout in seconds (default: 30)
- `GEMINI_TEMPERATURE` - Response temperature 0.0-1.0 (default: 0.7)
- `GEMINI_MAX_TOKENS` - Maximum response tokens (default: 2048)
- `GEMINI_RATE_LIMITING` - Enable rate limiting (default: true)
- `GEMINI_REQUESTS_PER_MINUTE` - Rate limit (default: 15)

### Optional Application Configuration
- `DEBUG_LOGGING` - Enable debug logs (default: false)
- `DATA_DIR` - Override data directory
- `MODELS_DIR` - Override models directory
- `RESULTS_EXPORT_DIR` - Override results directory

## Getting Your API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file
4. **Never share or commit this key to version control**

## Troubleshooting

### API Key Issues
- Ensure your API key starts with `AIza` and is exactly 39 characters
- Check that the `.env` file is in the same directory as the application
- Verify the API key has proper permissions in Google AI Studio

### Environment Loading Issues
- Check the application logs for environment configuration messages
- Ensure the `.env` file has no syntax errors
- Use the format: `KEY=value` (no spaces around =)

### Security Warnings
- If you see "No secure API key configured", check your `.env` file
- If API features are disabled, verify your API key is valid and active

## Best Practices

1. **Never commit `.env` files** to version control
2. **Use different API keys** for development and production
3. **Regularly rotate your API keys** for security
4. **Monitor API usage** in Google AI Studio
5. **Keep backups** of your configuration (without API keys)

## Migration from Old Config

If you have an existing `config.json` with an API key:

1. Copy the API key from `config.json`
2. Add it to your `.env` file as `GEMINI_API_KEY=your_key_here`
3. Remove the API key from `config.json` (set it to empty string)
4. The application will now use the secure environment variable

## Support

If you encounter issues with the security setup:

1. Check the application logs for detailed error messages
2. Verify your `.env` file format matches the example
3. Ensure your API key is valid and active
4. Contact support with log messages (never include your actual API key)