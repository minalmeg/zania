import json
import os
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

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

# Function to post a message to Slack
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
def download_file(file_url, headers):
    """Downloads a file from the given URL and saves it to the Dataset/user_data folder."""
    try:
        response = requests.get(file_url, headers=headers)

        # Create the directory if it does not exist
        save_dir = "Dataset/user_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Extract the file name and save path
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(save_dir, file_name)

        # Save the file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        logging.info(f"File downloaded and saved to: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        return None

def handle_message(event):
    """Handles message events when the bot is mentioned."""
    try:
        message_text = event.get('text')
        channel_id = event.get('channel')
        logging.info(f"Message received in channel {channel_id}: {message_text}")

        # Check if the bot is mentioned in the message
        if f"<@{bot_user_id}>" in message_text:
            post_to_slack(f"Bot mentioned! Message: {message_text}", channel_id)
    except Exception as e:
        logging.error(f"Error handling message: {e}")

def handle_file_upload(event):
    """Handles file upload events and acknowledges the upload."""
    try:
        file_id = event.get('file_id')
        file_info = client.files_info(file=file_id)
        file_url = file_info['file']['url_private']
        file_type = file_info['file']['mimetype']

        # Get the channel ID correctly from the event
        channel_id = event.get('channel_id')  # Extract the channel_id

        headers = {"Authorization": f"Bearer {bot_oauth_token}"}

        # Download the file and save it to the Dataset/user_data folder
        file_path = download_file(file_url, headers)

        if file_path:
            post_to_slack(f"File received and saved: {file_type}", channel_id)
        else:
            post_to_slack(f"File could not be saved.", channel_id)

        logging.info(f"File uploaded: {file_url} (type: {file_type})")
    except SlackApiError as e:
        logging.error(f"Error handling file upload: {e.response['error']}")
    except Exception as e:
        logging.error(f"Error in file handling: {e}")

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

            # Handle messages in the channel where the bot is mentioned
            if event.get('type') == 'message' and 'subtype' not in event:
                handle_message(event)

            # Handle file uploads and save the file in the Dataset/user_data folder
            elif event.get('type') == 'file_shared':
                handle_file_upload(event)

        return jsonify({'status': 'ok'})

    except Exception as e:
        logging.error(f"Error handling event: {e}")
        return "Internal Server Error", 500

# Start the Flask server
if __name__ == "__main__":
    app.run(port=8080)
