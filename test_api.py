import json
import os
import requests
import csv
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
from rag_processor import process_pdf_and_questions

app = Flask(__name__)

# Set up logging to a file
log_file = "slack_logging.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        # logging.StreamHandler()  # This will keep printing to console too, can be removed if you only want file
    ]
)

# Function to load keys and secrets from the config.json file
def load_config(file_path):
    """Loads the configuration from a JSON file."""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load the config.json file
config_file_path = "/home/minal/Desktop/zania/config.json"  # Replace with the path to your config file
config = load_config(config_file_path)

# Extract the Slack tokens
bot_oauth_token = config["slack"]["bot_oauth_token"]

# Initialize the Slack client with the bot OAuth token
client = WebClient(token=bot_oauth_token)

# Fetch bot user ID to check for mentions later
bot_user_id = client.auth_test()["user_id"]

# CSV file to maintain original and new file names and user text
csv_file_path = "Dataset/user_data/file_mapping.csv"

def initialize_csv_file():
    """Initializes the CSV file if it doesn't exist."""
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['original_name', 'new_name', 'user_text'])  # Write the header
        logging.info(f"CSV file created at {csv_file_path}")

def update_csv_file(original_name, new_name, user_text):
    """Updates the CSV file with the original and new file names, and the user text."""
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([original_name, new_name, user_text])
    logging.info(f"File mapping updated: {original_name} -> {new_name}, with text: {user_text}")

# Function to get the next available file number
def get_next_file_number(save_dir):
    """Returns the next file number for naming convention."""
    existing_files = [f for f in os.listdir(save_dir) if f.startswith('datafile_') and f.endswith('.pdf')]
    if not existing_files:
        return 0  # If no files, start with 0
    else:
        numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        return max(numbers) + 1  # Increment by 1 from the largest number

def extract_text_from_event(event):
    """Extracts the user-sent message from the Slack event payload, even in file_shared events."""
    try:
        # Extract from the 'text' field directly
        if 'text' in event:
            user_text = event['text']
            # Remove bot mention (e.g., <@U07N7LRM3EH>)
            user_text = ' '.join([word for word in user_text.split() if not word.startswith('<@')])
            logging.info(f"Extracted user text from 'text': {user_text}")
            return user_text.strip()

        logging.error("No 'text' field found in the event.")
        return ""
    except (KeyError, IndexError) as e:
        logging.error(f"Error extracting text from event: {e}")
        return ""

# Define post_to_slack function to send responses back to Slack
def post_to_slack(message, channel_id):
    """Posts a message to a Slack channel."""
    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        logging.info(f"Message posted: {response['ts']}")
    except SlackApiError as e:
        logging.error(f"Error posting to Slack: {e.response['error']}")

# Function to download a file from Slack and save it in Dataset/user_data
def download_file(file_url, headers, original_name, user_text):
    """Downloads a file from the given URL, renames it, and saves it in Dataset/user_data."""
    try:
        response = requests.get(file_url, headers=headers)

        # Create the directory if it does not exist
        save_dir = "Dataset/user_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the next file number for naming convention
        next_file_number = get_next_file_number(save_dir)
        new_file_name = f"datafile_{next_file_number:04d}.pdf"
        file_path = os.path.join(save_dir, new_file_name)

        # Save the file with the new name
        with open(file_path, 'wb') as file:
            file.write(response.content)

        # Log the download and update the CSV file with the original and new file names, and user text
        logging.info(f"File downloaded and saved to: {file_path}")
        logging.info(f"Saving to CSV: {original_name}, {new_file_name}, {user_text}")  # Log the data before saving
        update_csv_file(original_name, new_file_name, user_text)

        return file_path
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        return None

def handle_message(event):
    """Handles message events when the bot is mentioned."""
    try:
        # Check if the message is from the bot itself (prevent spam)
        if 'bot_id' in event:
            logging.info(f"Ignoring bot's own message to prevent spam: {event.get('text')}")
            return

        message_text = event.get('text')
        channel_id = event.get('channel')
        logging.info(f"Message received in channel {channel_id}: {message_text}")

        # Check if the bot is mentioned in the message
        if f"<@{bot_user_id}>" in message_text:
            post_to_slack(f"Bot mentioned! Message: {message_text}", channel_id)
    except Exception as e:
        logging.error(f"Error handling message: {e}")

def handle_file_upload(event):
    """Handles file upload events, checks file type, and acknowledges the upload."""
    try:
        # Check if 'files' are present in the event
        if 'files' not in event or not event['files']:
            # No file attached, post message to Slack and stop further processing
            channel_id = event.get('channel')
            post_to_slack("Please attach a PDF file.", channel_id)
            return

        file_info = event['files'][0]  # Access the first file info (if multiple files, handle the first)
        file_url = file_info['url_private']
        original_file_name = file_info['name']
        file_type = file_info['mimetype']
        channel_id = event.get('channel')

        headers = {"Authorization": f"Bearer {bot_oauth_token}"}

        # Check if the file is a PDF, otherwise return an error message and stop further processing
        if file_type != 'application/pdf':
            post_to_slack("Only PDF files are accepted. Please upload a PDF.", channel_id)
            return

        # Extract user text from event
        user_text = extract_text_from_event(event)

        # Check if the CSV already contains this file and text
        if is_duplicate_entry(original_file_name, user_text):
            logging.info(f"Duplicate entry detected: {original_file_name}, {user_text}")
            return

        # Download the file, rename it, and save it in the Dataset/user_data folder
        file_path = download_file(file_url, headers, original_file_name, user_text)

        response = process_pdf_and_questions(file_path, user_text)
        print(response)
        if file_path:
            post_to_slack(f"File received and saved: {file_type}", channel_id)
            post_to_slack(f"Answer: {response}", channel_id)
        else:
            post_to_slack("File could not be saved.", channel_id)

        logging.info(f"File uploaded: {file_url} (type: {file_type})")
    except SlackApiError as e:
        logging.error(f"Error handling file upload: {e.response['error']}")
    except Exception as e:
        logging.error(f"Error in file handling: {e}")


def is_duplicate_entry(original_name, user_text):
    """Checks if the combination of original_name and user_text already exists in the CSV."""
    try:
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) >= 3 and row[0] == original_name and row[2] == user_text:
                    return True
    except FileNotFoundError:
        logging.warning("CSV file not found when checking for duplicates.")
    return False

@app.route('/slack/events', methods=['GET', 'POST'])
def slack_events():
    """Handles incoming Slack events."""
    if request.method == 'GET':
        return "This endpoint expects POST requests from Slack.", 405  # For browsers, return a message

    try:
        data = request.json
        logging.info(f"Event received: {data}")

        if 'challenge' in data:
            # Respond to the Slack URL verification challenge
            return jsonify({'challenge': data['challenge']})

        if 'event' in data:
            event = data['event']

            # Handle file uploads first and avoid duplicate message handling
            if 'files' in event:
                handle_file_upload(event)
            # Handle messages in the channel where the bot is mentioned
            elif event.get('type') == 'message' and 'subtype' not in event:
                handle_message(event)

        return jsonify({'status': 'ok'})

    except Exception as e:
        logging.error(f"Error handling event: {e}")
        return "Internal Server Error", 500

# Start the Flask server
if __name__ == "__main__":
    # Ensure CSV file exists
    initialize_csv_file()
    app.run(port=8080)