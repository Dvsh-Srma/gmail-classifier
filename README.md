# Gmail Classifier

An intelligent email classification system that uses Google's Gemini AI to automatically categorize Gmail messages into a structured label hierarchy.

## Features

- 🤖 AI-powered email classification using Google's Gemini AI
- 📧 Automatic Gmail label management
- 🏷️ Hierarchical label structure for organized email management
- ⚡ Processes recent emails based on configurable time window
- 🔒 Secure credential management
- 📝 Detailed logging for monitoring and debugging

## Prerequisites

- Python 3.7+
- Gmail account
- Google Cloud Project with Gmail API enabled
- Gemini API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dvsh-Srma/gmail-classifier.git
cd gmail-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud Project:
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials and save as `credentials.json`

4. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. First-time setup:
   - Place your `credentials.json` file in the project root
   - Run the script - it will open a browser window for Gmail authentication
   - The authentication token will be saved in `token.pickle`

2. Run the classifier:
```bash
python add_labels_in_gmail.py
```

By default, the script processes emails from the last 2 days. You can modify this in the script.

## Label Structure

The system uses a hierarchical label structure:

- ⚡ Action-Required/
  - Urgent
  - Follow-Up
  - To-Do
  - Waiting-On
- 💼 Projects-and-Work/
  - Client-A/
    - In-Progress
    - Completed
  - Internal/
    - Team
    - Announcements
- 🏠 Personal-Life/
  - Family
  - Friends
  - Finances/
    - Bills-and-Payments
    - Investments
  - Travel/
    - Destinations
  - Shopping/
    - Receipts
- 📚 Reference/
  - Learning/
    - Courses
    - Webinars
  - Documents/
    - Contracts
- 📬 Newsletters/
  - Tech-Updates
  - Personal-Development
- 🌍 Social-and-Community/
  - Forums
  - Group-Updates
- 🗑️ Other/
  - Maybe-Delete

## Configuration

- Modify `LABELS` in the script to customize the label structure
- Adjust `MAX_CONTENT_TOKENS` to control how much email content is sent to Gemini
- Change `API_CALL_DELAY_SECONDS` to manage API rate limits

## Security Notes

- Never commit `credentials.json`, `token.pickle`, or `.env` files
- Keep your API keys secure
- The `.gitignore` file is configured to exclude sensitive files

## Logging

Logs are written to `gmail_classifier.log` and include:
- Email processing status
- Classification results
- Error messages
- API interactions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
