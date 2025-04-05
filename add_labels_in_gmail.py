import os
import pickle
import base64
import json
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Google API Imports ---
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import google.auth

# --- Third-party Imports ---
import requests
from tiktoken import encoding_for_model, Encoding # Use the specific class for type hint

# --- Configuration ---
# Credentials and Token Files
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# Gemini API Configuration (Use environment variables!)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Load from environment
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    # Consider raising an exception or exiting if the key is mandatory
    # exit(1)
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FALLBACK" # Or use a fallback for testing, but not recommended for production

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
GEMINI_MODEL = "gemini-pro" # Model used for token counting and API calls
MAX_CONTENT_TOKENS = 30000 # Max tokens to send to Gemini (adjust based on model limits & desired context)
API_CALL_DELAY_SECONDS = 2 # Delay between Gemini API calls to avoid rate limits

# Gmail API Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
GMAIL_USER_ID = 'me' # Standard user ID for Gmail API

# --- Label Structure ---
# Define your desired label structure here. This list will be used for:
# 1. Creating labels in Gmail if they don't exist.
# 2. Providing the list of valid labels to the Gemini model in the prompt.
LABELS = [
    "⚡ Action-Required/Urgent",
    "🕓 Action-Required/Follow-Up",
    "📥 Action-Required/To-Do",
    "⏳ Action-Required/Waiting-On",

    "💼 Projects-and-Work/Client-A/🚧 In-Progress",
    "💼 Projects-and-Work/Client-A/✅ Completed",
    "💼 Projects-and-Work/Internal/👥 Team",
    "💼 Projects-and-Work/Internal/📢 Announcements",

    "🏠 Personal-Life/Family",
    "🏠 Personal-Life/Friends",
    "🏠 Personal-Life/Finances/💸 Bills-and-Payments",
    "🏠 Personal-Life/Finances/📈 Investments",
    "🏠 Personal-Life/Travel/🌍 Destinations",
    "🏠 Personal-Life/Shopping/🧾 Receipts",

    "📚 Reference/Learning/📖 Courses",
    "📚 Reference/Learning/🧠 Webinars",
    "📚 Reference/Documents/📄 Contracts",

    "📬 Newsletters/📰 Tech-Updates",
    "📬 Newsletters/💡 Personal-Development",

    "🌍 Social-and-Community/💬 Forums",
    "🌍 Social-and-Community/📢 Group-Updates",

    "🗑️ Other/Maybe-Delete", # Added an example fallback/other label
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("gmail_classifier.log"), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR) # Silence noisy google cache logs

# --- Helper Functions ---

def count_tokens(text: str, model: str = GEMINI_MODEL) -> int:
    """Counts the number of tokens in a given text using tiktoken."""
    try:
        encoding: Encoding = encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        return len(text.split()) # Fallback to word count approximation

def clean_email_content(text: str) -> str:
    """Removes HTML tags and excessive whitespace from email content."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove style blocks first
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE) # Remove script blocks
    text = re.sub(r'<.*?>', ' ', text, flags=re.DOTALL) # Remove remaining HTML tags
    # Remove URLs (optional, uncomment if desired)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Replace multiple whitespace characters (including newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_text_by_tokens(text: str, max_tokens: int, model: str = GEMINI_MODEL) -> str:
    """Truncates text to fit within the specified token limit."""
    token_count = count_tokens(text, model)
    if token_count <= max_tokens:
        return text

    logging.warning(f"Content length ({token_count} tokens) exceeds limit ({max_tokens}). Truncating.")
    # Simple truncation for now - might lose context at the end.
    # A more sophisticated approach could try to summarize or keep beginning/end.
    try:
        encoding: Encoding = encoding_for_model(model)
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        # Add an indicator that the text was truncated
        return truncated_text + " ... [Content Truncated]"
    except Exception as e:
        logging.error(f"Error during token-based truncation: {e}. Falling back to character-based.")
        # Fallback: approximate character limit based on average token length (crude)
        avg_chars_per_token = len(text) / token_count if token_count > 0 else 4
        estimated_char_limit = int(max_tokens * avg_chars_per_token * 0.9) # 90% buffer
        return text[:estimated_char_limit] + " ... [Content Truncated]"

def get_email_body(payload: Dict[str, Any]) -> Optional[str]:
    """Extracts and decodes the email body from the message payload."""
    body = None
    if 'parts' in payload:
        parts = payload['parts']
        # Prioritize text/plain, then text/html
        plain_part = next((part for part in parts if part['mimeType'] == 'text/plain'), None)
        html_part = next((part for part in parts if part['mimeType'] == 'text/html'), None)

        target_part = None
        if plain_part:
            target_part = plain_part
        elif html_part:
            target_part = html_part
            logging.debug("Using HTML part as plain text part is missing.")
        # Handle nested parts (e.g., multipart/alternative)
        elif parts[0].get('parts'):
             # Recurse into nested parts, commonly found in multipart/alternative
             nested_plain = next((p for p in parts[0]['parts'] if p['mimeType'] == 'text/plain'), None)
             nested_html = next((p for p in parts[0]['parts'] if p['mimeType'] == 'text/html'), None)
             if nested_plain:
                 target_part = nested_plain
             elif nested_html:
                 target_part = nested_html
                 logging.debug("Using nested HTML part.")

        if target_part and 'body' in target_part and 'data' in target_part['body']:
            data = target_part['body']['data']
            try:
                # Replace URL-safe base64 characters and decode
                byte_code = base64.urlsafe_b64decode(data.replace('-', '+').replace('_', '/'))
                # Try decoding with common encodings
                for encoding_type in ['utf-8', 'iso-8859-1', 'windows-1252']:
                    try:
                        body = byte_code.decode(encoding_type)
                        logging.debug(f"Decoded body using {encoding_type}")
                        break # Success
                    except UnicodeDecodeError:
                        continue # Try next encoding
                if body is None:
                    logging.warning("Could not decode email body with common encodings.")
            except base64.binascii.Error as e:
                logging.error(f"Base64 decoding error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error decoding body: {e}")
        elif payload.get('body') and payload['body'].get('data'):
             # Sometimes the body is directly in the main payload (simple emails)
             data = payload['body']['data']
             try:
                byte_code = base64.urlsafe_b64decode(data.replace('-', '+').replace('_', '/'))
                for encoding_type in ['utf-8', 'iso-8859-1', 'windows-1252']:
                    try:
                        body = byte_code.decode(encoding_type)
                        logging.debug(f"Decoded body directly from payload using {encoding_type}")
                        break
                    except UnicodeDecodeError:
                        continue
                if body is None:
                     logging.warning("Could not decode direct email body.")
             except base64.binascii.Error as e:
                logging.error(f"Base64 decoding error (direct payload): {e}")
             except Exception as e:
                logging.error(f"Unexpected error decoding direct body: {e}")

    if body and html_part and not plain_part: # If we only had HTML
        logging.debug("Cleaning HTML content before returning.")
        body = clean_email_content(body) # Clean HTML if it was the only source

    elif not body:
        logging.warning("Could not extract readable body content.")

    return body

# --- Gmail API Functions ---

def get_gmail_service() -> Optional[Resource]:
    """Authenticates and builds the Gmail API service client."""
    creds = None
    if not os.path.exists(CREDENTIALS_FILE):
        logging.error(f"Credentials file not found at: {CREDENTIALS_FILE}")
        print(f"Error: Credentials file '{CREDENTIALS_FILE}' not found.")
        print("Please download your credentials from Google Cloud Console and place it in the same directory.")
        return None

    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
            logging.info("Loaded credentials from token file.")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
             logging.error(f"Error loading token file: {e}. Will re-authenticate.")
             creds = None # Force re-authentication

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logging.info("Refreshing expired credentials.")
                creds.refresh(Request())
            except Exception as e:
                logging.error(f"Failed to refresh token: {e}. Need to re-authenticate.")
                # Delete potentially corrupted token file if refresh fails
                if os.path.exists(TOKEN_FILE):
                     try:
                         os.remove(TOKEN_FILE)
                         logging.info(f"Removed potentially corrupt token file: {TOKEN_FILE}")
                     except OSError as remove_err:
                         logging.error(f"Error removing token file {TOKEN_FILE}: {remove_err}")
                creds = None # Force re-authentication flow
        else:
            logging.info("No valid credentials found, initiating OAuth flow.")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                # Specify a fixed port or let it choose dynamically (port=0)
                creds = flow.run_local_server(port=0)
                logging.info("OAuth flow completed successfully.")
            except FileNotFoundError:
                logging.error(f"Credentials file not found during OAuth flow: {CREDENTIALS_FILE}")
                return None
            except Exception as e:
                logging.error(f"Error during OAuth flow: {e}")
                return None
        # Save the credentials for the next run
        try:
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
            logging.info(f"Saved new credentials to {TOKEN_FILE}")
        except Exception as e:
            logging.error(f"Error saving token file: {e}")

    if not creds:
        logging.error("Failed to obtain credentials.")
        return None

    try:
        service = build('gmail', 'v1', credentials=creds)
        logging.info("Gmail API service built successfully.")
        return service
    except Exception as e:
        logging.error(f"Failed to build Gmail service: {e}")
        return None

def create_labels_if_needed(service: Resource) -> Dict[str, str]:
    """Checks existing Gmail labels and creates any missing ones defined in LABELS."""
    existing_labels_map: Dict[str, str] = {}
    try:
        results = service.users().labels().list(userId=GMAIL_USER_ID).execute()
        existing_labels = results.get('labels', [])
        existing_labels_map = {label['name']: label['id'] for label in existing_labels}
        logging.info(f"Found {len(existing_labels_map)} existing labels.")
        logging.debug(f"Existing labels: {list(existing_labels_map.keys())}")

        created_count = 0
        for label_name in LABELS:
            if label_name not in existing_labels_map:
                logging.info(f"Label '{label_name}' not found. Creating...")
                label_body = {
                    'name': label_name,
                    'messageListVisibility': 'show',
                    'labelListVisibility': 'labelShow'
                    # Consider adding color options if desired:
                    # 'color': {'textColor': '#ffffff', 'backgroundColor': '#000000'}
                }
                try:
                    created_label = service.users().labels().create(userId=GMAIL_USER_ID, body=label_body).execute()
                    existing_labels_map[label_name] = created_label['id'] # Add newly created label to map
                    logging.info(f"Successfully created label '{label_name}' with ID {created_label['id']}")
                    created_count += 1
                except HttpError as e:
                    logging.error(f"Failed to create label '{label_name}': {e}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred creating label '{label_name}': {e}")

        if created_count > 0:
             logging.info(f"Created {created_count} new labels.")
        else:
             logging.info("All required labels already exist.")

    except HttpError as e:
        logging.error(f"Failed to list labels: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred checking/creating labels: {e}")

    return existing_labels_map # Return the map including any newly created labels


def update_email_label(service: Resource, message_id: str, new_label_name: str, all_labels_map: Dict[str, str]) -> bool:
    """Removes old classification labels and applies the new one to an email."""
    if not new_label_name or new_label_name not in all_labels_map:
        logging.warning(f"Invalid or missing new label name ('{new_label_name}') for message {message_id}. Skipping update.")
        return False

    new_label_id = all_labels_map[new_label_name]
    labels_to_remove_ids = []

    try:
        # Get existing labels on the message
        msg = service.users().messages().get(userId=GMAIL_USER_ID, id=message_id, format='metadata', metadataHeaders=['labelIds']).execute()
        current_label_ids = msg.get('labelIds', [])

        # Identify which of *our managed* labels are currently on the message
        for label_name, label_id in all_labels_map.items():
            # Check if the label is one we manage (from LABELS list) and is currently applied
            if label_name in LABELS and label_id in current_label_ids:
                 # Don't remove the label if it's the one we want to add
                if label_id != new_label_id:
                    labels_to_remove_ids.append(label_id)
                    logging.debug(f"Identified label to remove: '{label_name}' (ID: {label_id}) from message {message_id}")

        # Prepare the modification request
        modify_body = {}
        if labels_to_remove_ids:
            modify_body['removeLabelIds'] = labels_to_remove_ids
            logging.info(f"Preparing to remove {len(labels_to_remove_ids)} old labels from message {message_id}.")

        # Only add the new label if it's not already present (though modify usually handles this)
        if new_label_id not in current_label_ids:
             modify_body['addLabelIds'] = [new_label_id]
             logging.info(f"Preparing to add label '{new_label_name}' (ID: {new_label_id}) to message {message_id}.")
        elif not labels_to_remove_ids:
             logging.info(f"Label '{new_label_name}' is already correctly applied to message {message_id}. No changes needed.")
             return True # Already correctly labelled


        # Execute the modification if there's anything to change
        if modify_body:
             logging.debug(f"Executing modify request for message {message_id}: {modify_body}")
             service.users().messages().modify(userId=GMAIL_USER_ID, id=message_id, body=modify_body).execute()
             logging.info(f"Successfully updated labels for message {message_id}. Added: '{new_label_name}', Removed IDs: {labels_to_remove_ids}")
             return True
        else:
             # This case should ideally be caught above, but added for safety.
             logging.info(f"No label modifications needed for message {message_id}.")
             return True

    except HttpError as e:
        # Handle common errors like label not found (shouldn't happen with our checks) or insufficient permissions
        logging.error(f"HttpError updating labels for message {message_id}: {e}")
        # Check if the error is due to the message being deleted/moved
        if e.resp.status == 404:
            logging.warning(f"Message {message_id} not found. It might have been deleted or moved.")
            return False # Indicate failure due to message not found
        return False
    except Exception as e:
        logging.error(f"Unexpected error updating label for message {message_id}: {e}")
        return False


# --- Gemini API Function ---

def classify_email_with_gemini(sender: str, subject: str, content: str, email_date: str) -> Optional[str]:
    """
    Classifies an email using the Gemini API based on sender, subject, and content.

    Args:
        sender: Email sender address/name.
        subject: Email subject line.
        content: Cleaned and potentially truncated email body content.
        email_date: Date the email was received.

    Returns:
        The suggested label name string, or None if classification fails.
    """
    logging.debug(f"Classifying email - Subject: '{subject}', Sender: '{sender}', Date: {email_date}")

    if not content and not subject: # Cannot classify without content or subject
        logging.warning("Cannot classify email with empty subject and content.")
        return None

    # --- Construct the Prompt ---
    # Dynamically create the list of labels for the prompt
    label_list_string = "\n".join([f"- {label}" for label in LABELS])

    prompt = f"""
You are an intelligent email classifier for Gmail. Your goal is to assign the single most appropriate label to the following email based on its sender, subject, content, and date.

**Available Labels:**
Please choose ONLY ONE label from the following list:
{label_list_string}

**Email Details:**
- **Sender:** {sender}
- **Subject:** {subject}
- **Date Received:** {email_date}
- **Content Snippet:**
---
{content}
---

**Instructions:**
1. Analyze the sender, subject, content, and date.
2. Select the single most fitting label from the "Available Labels" list provided above.
3. Respond ONLY with a JSON object containing the selected label under the key "label".

**Response Format (Strict JSON):**
{{{{
  "label": "Your Selected Label Here"
}}}}

Do NOT add any explanation, commentary, or introductory text before or after the JSON object. Only return the JSON. If no label seems appropriate, you can use "🗑️ Other/Maybe-Delete" if it's in the list, otherwise, make the best possible choice from the available list.
"""

    # --- Prepare API Request ---
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        # Optional: Configure safety settings if needed
        # "safetySettings": [
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     # Add other categories as needed
        # ],
        "generationConfig": {
            "temperature": 0.2, # Lower temperature for more deterministic classification
            "maxOutputTokens": 100, # Should be enough for just the JSON label
            "topP": 0.9,
            "topK": 10
        }
    }

    logging.debug(f"Sending request to Gemini API. Prompt length: {len(prompt)} chars, Content Tokens (approx): {count_tokens(content)}")

    try:
        # --- Make API Call ---
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=60) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # --- Process Response ---
        response_json = response.json()
        logging.debug(f"Raw Gemini API Response: {response_json}")

        # Extract the text content from the response
        if 'candidates' in response_json and response_json['candidates']:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                generated_text = candidate['content']['parts'][0].get('text', '').strip()

                # Attempt to parse the JSON from the response text
                try:
                    # Sometimes the model might wrap the JSON in markdown backticks
                    cleaned_text = generated_text.strip().removeprefix('```json').removesuffix('```').strip()
                    parsed_json = json.loads(cleaned_text)

                    if isinstance(parsed_json, dict) and 'label' in parsed_json:
                        suggested_label = parsed_json['label']
                        # Validate if the suggested label is in our defined list
                        if suggested_label in LABELS:
                            logging.info(f"Gemini suggested label: '{suggested_label}'")
                            return suggested_label
                        else:
                            logging.warning(f"Gemini suggested label '{suggested_label}' is not in the predefined LABELS list. Attempting best match or ignoring.")
                            # Optional: Implement fuzzy matching here if desired
                            # For now, we treat it as invalid.
                            return None # Or return a default like "Other"
                    else:
                         logging.error(f"Gemini response JSON does not contain the 'label' key. Response: {cleaned_text}")
                         return None

                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON from Gemini response: {generated_text}")
                    # Optional: Try to extract label with regex as a fallback? Risky.
                    return None
            else:
                 logging.error("Invalid Gemini response structure: Missing 'content' or 'parts'.")
                 return None
        elif 'promptFeedback' in response_json:
             # Handle cases where the prompt was blocked
             block_reason = response_json['promptFeedback'].get('blockReason')
             safety_ratings = response_json['promptFeedback'].get('safetyRatings')
             logging.error(f"Gemini request blocked. Reason: {block_reason}. Ratings: {safety_ratings}")
             return None
        else:
            logging.error(f"Unexpected Gemini response format: {response_json}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request to Gemini API failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini API call: {e}")
        return None
    finally:
        # Add delay before next API call
        logging.debug(f"Waiting for {API_CALL_DELAY_SECONDS} seconds before next API call...")
        time.sleep(API_CALL_DELAY_SECONDS)


# --- Main Processing Function ---

def process_recent_emails(days: int = 5) -> None:
    """
    Fetches emails from the last 'days', classifies them using Gemini,
    and applies the corresponding label.
    """
    logging.info(f"Starting email processing for the last {days} days.")

    # 1. Authenticate and get Gmail Service
    service = get_gmail_service()
    if not service:
        logging.critical("Failed to get Gmail service. Exiting.")
        return

    # 2. Ensure all required labels exist
    all_labels_map = create_labels_if_needed(service)
    if not all_labels_map:
         logging.warning("Label map is empty, possibly due to errors listing/creating labels. Processing might fail.")
         # Decide if you want to continue or exit
         # return

    # 3. Define Search Query
    try:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
        # Query for unread emails in inbox/all mail, not in Spam/Trash, after start_date
        # Modify query as needed. Examples:
        # 'in:inbox is:unread after:{start_date}' # Only unread in inbox
        # 'after:{start_date} -in:spam -in:trash' # All mail except spam/trash
        query = f'in:inbox after:{start_date} -in:spam -in:trash'
        logging.info(f"Using query: '{query}'")

        # 4. List Messages
        all_message_ids = []
        page_token = None
        while True:
            logging.info(f"Fetching message list..." + (f" (Page Token: {page_token})" if page_token else ""))
            list_request = service.users().messages().list(userId=GMAIL_USER_ID, q=query, pageToken=page_token, maxResults=100) # Fetch 100 at a time
            response = list_request.execute()
            messages = response.get('messages', [])
            if messages:
                all_message_ids.extend([msg['id'] for msg in messages])
                logging.info(f"Fetched {len(messages)} message IDs. Total so far: {len(all_message_ids)}")
            else:
                 logging.info("No messages found matching the query in this page.")

            page_token = response.get('nextPageToken')
            if not page_token:
                logging.info("Finished fetching all message IDs.")
                break # Exit loop when no more pages

    except HttpError as e:
        logging.error(f"Failed to list messages: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching message list: {e}")
        return

    if not all_message_ids:
        logging.info("No emails found matching the criteria. Processing finished.")
        return

    logging.info(f"Found {len(all_message_ids)} emails to potentially process.")

    # 5. Process Each Message
    processed_count = 0
    labeled_count = 0
    failed_count = 0
    skipped_count = 0

    for i, message_id in enumerate(all_message_ids):
        logging.info(f"--- Processing email {i + 1}/{len(all_message_ids)} (ID: {message_id}) ---")
        try:
            # Get full message details
            # Using 'full' format is easier but fetches more data than potentially needed.
            # Alternatively use 'metadata' for headers and 'raw' for body if optimizing bandwidth.
            msg_request = service.users().messages().get(userId=GMAIL_USER_ID, id=message_id, format='full')
            message = msg_request.execute()

            payload = message.get('payload', {})
            headers = payload.get('headers', [])

            # Extract relevant headers
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '[No Subject]')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '[No Sender]')
            date_str = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')

            # Parse date string (handle potential formatting issues)
            email_date = "Unknown Date"
            if date_str:
                try:
                    # Example: 'Tue, 1 Apr 2025 10:30:00 +0530 (IST)' or '1 Apr 2025 10:30:00 +0530'
                    # Python's datetime parsing can be tricky with timezones. dateutil library is more robust if needed.
                    # Simple attempt:
                    email_date_dt = datetime.strptime(date_str.split(' (')[0].split(' +')[0].split(' -')[0].strip(), '%a, %d %b %Y %H:%M:%S')
                    email_date = email_date_dt.strftime('%Y-%m-%d')
                except ValueError:
                    logging.warning(f"Could not parse date string: {date_str}. Using original.")
                    email_date = date_str # Fallback to original string

            # Extract and clean body
            raw_body = get_email_body(payload)
            if not raw_body:
                 logging.warning(f"Could not extract body for message {message_id}. Skipping classification based on content.")
                 # Decide: skip entirely, or try classifying based on subject/sender only?
                 # For now, skip full classification if body is missing.
                 skipped_count += 1
                 continue # Skip this email if body extraction failed

            # Clean the body (remove HTML, etc.)
            cleaned_body = clean_email_content(raw_body)

            # Truncate if necessary before sending to Gemini
            truncated_body = truncate_text_by_tokens(cleaned_body, MAX_CONTENT_TOKENS, GEMINI_MODEL)

            # Classify using Gemini
            suggested_label = classify_email_with_gemini(sender, subject, truncated_body, email_date)

            # Apply the label
            if suggested_label:
                success = update_email_label(service, message_id, suggested_label, all_labels_map)
                if success:
                    labeled_count += 1
                else:
                    failed_count += 1 # Label update failed
            else:
                logging.warning(f"Could not classify email ID {message_id} (Subject: '{subject}'). Skipping labeling.")
                skipped_count += 1 # Classification failed

            processed_count += 1

        except HttpError as e:
            logging.error(f"HttpError processing message ID {message_id}: {e}")
            # Check if the error is due to the message being deleted/moved
            if e.resp.status == 404:
                logging.warning(f"Message {message_id} not found during processing. It might have been deleted or moved.")
            failed_count += 1
        except Exception as e:
            logging.error(f"Unexpected error processing message ID {message_id}: {e}", exc_info=True) # Log traceback
            failed_count += 1
        finally:
             logging.info(f"--- Finished processing email {message_id} ---")


    logging.info("=" * 30)
    logging.info("Email Processing Summary:")
    logging.info(f"  Total emails found:      {len(all_message_ids)}")
    logging.info(f"  Emails processed:        {processed_count}")
    logging.info(f"  Emails successfully labeled: {labeled_count}")
    logging.info(f"  Emails skipped/unlabeled: {skipped_count}")
    logging.info(f"  Emails failed processing: {failed_count}")
    logging.info("=" * 30)

# --- Main Execution Guard ---

if __name__ == '__main__':
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK":
         logging.warning("GEMINI_API_KEY is not set or is using the fallback value. Classification will likely fail.")
         print("\nWARNING: Gemini API Key is not configured correctly. Please set the GEMINI_API_KEY environment variable.\n")
         # Optional: exit if key is absolutely required
         # exit(1)

    # Set how many days back you want to process emails
    days_to_process = 2 # Example: Process emails from the last 2 days
    logging.info(f"Starting script execution. Processing emails from the last {days_to_process} days.")
    process_recent_emails(days=days_to_process)
    logging.info("Script execution finished.")