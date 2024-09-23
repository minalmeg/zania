# RAG Processor

This project processes PDFs using Retrieval-Augmented Generation (RAG) with OpenAI's models and FAISS for efficient text retrieval.

## Setup

### RAG Code setup

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/rag-processor.git
   cd rag-processor
   ```

2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Download spaCy model**:
    ```
    python -m spacy download en_core_web_sm
    ```

4. **Set up OpenAI API key**:
    Create a config.json file with your OpenAI API key:
    ```
    {
        "openai": {
            "api_key": "your-openai-api-key"
        }
    }
    ```

### Slack API Setup

1. **Install ngrok**:
   - Download and install `ngrok` from [ngrok's official site](https://ngrok.com/download).
   - Unzip the downloaded file and run the following command to install it:
     ```bash
     ./ngrok authtoken YOUR_NGROK_AUTH_TOKEN
     ```

2. **Start ngrok**:
   - Run `ngrok` to expose your local server to the internet:
     ```bash
     ngrok http 8080
     ```
   - Note the generated public URL (e.g., `https://your-ngrok-url.ngrok.io`), which you will use in the Slack app configuration.

3. **Create a Slack App**:
   - Go to [Slack API](https://api.slack.com/apps) and create a new app.
   - Choose "From scratch" and provide a name for your app.
   - Select the workspace where you want to install the app.

4. **Enable OAuth & Permissions**:
   - Under "OAuth & Permissions," add the necessary scopes:
     - `chat:write` (for posting messages)
     - `files:read` (for reading uploaded files)
     - `files:write` (for writing files)
   - Save changes and install the app to your workspace.
   - Once installed, you will receive an OAuth Token. Copy this token and add it to your `config.json` file:
     ```json
     {
       "slack": {
         "bot_oauth_token": "YOUR_SLACK_OAUTH_TOKEN"
       }
     }
     ```

5. **Set up Event Subscriptions**:
   - Go to "Event Subscriptions" in the Slack app settings.
   - Enable event subscriptions by toggling the "Enable Events" switch.
   - Add the following Request URL using your `ngrok` public URL:
     ```
     https://your-ngrok-url.ngrok.io/slack/events
     ```
   - Subscribe to the necessary bot events such as:
     - `app_mention`
     - `file_shared`

6. **Add App to a Channel**:
   - Open Slack, and in the channel where you want the bot to be active, type:
     ```
     /invite @YourAppName
     ```
   - This will invite your app to the channel, allowing it to respond to mentions and file uploads.

7. **Get Channel ID**:
   - To get the Channel ID, go to the Slack workspace in a browser.
   - Click on the channel where you added the bot.
   - The Channel ID will be in the URL, formatted like this: `https://app.slack.com/client/TXXXXX/CXXXXXX`.
   - Copy the Channel ID (`CXXXXXX`) and add it to your `test_api.py` code:
     ```python
     channel_id = "CXXXXXX"
     ```

8. **Install PDF Extraction App**:
   - Go to your Slack app and click "Install App" in the app settings.
   - Add the app to the desired workspace by following Slack's installation instructions.

9. **Set up your Flask server**:
   - Ensure your local Flask server is running and listening to the `/slack/events` endpoint. You can use the following command to start your server:
     ```bash
     python app.py
     ```
   - Ensure your Flask app has the `/slack/events` route to handle incoming Slack events.

10. **Verify events**:
    - Slack will send a verification request to your `ngrok` URL. Make sure to handle this verification in your Flask app by responding with the challenge token provided in the request.
    - Once the verification is successful, your Slack app will be able to receive and process events such as file uploads and mentions.


## Running

1. **Run the API code**:
    ```bash
    python test_api.py
    ```

2. **Call from Slack**:
    ![Slack Call GIF](https://github.com/minalmeg/zania/blob/main/images/slack_call.gif)


## Improvements


### Accuracy Improvements

- Currently, a very basic code for checking the correctness of the answer is used. We can further improve this by generating a eval dataset and running a cosine similarity functionon the model to filter low confidence answers.

- Use the feedback to fine-tune the AI model. For instance, when a user marks an answers with thumbs up or down emoji, the system could flag the chunk(s) of text that were used to generate the answer. Over time, this data can be used to train a classifier that predicts whether a given chunk is likely to lead to accurate or inaccurate answers.

- Finetuning on domain specific data. In this example, it could be finetuning the model on The Ultimate Guide to POSH Compliance to integrate PoSH in any companies existing HR Policies documentation. 

### Scalibility Improvements

- Split large PDF documents into multiple chunks and process them concurrently using multi-threading or multi-processing.

- Use asynchronous methods for API calls (e.g., calling OpenAI or external services) to handle multiple requests in parallel without blocking.

### Production-Grade Improvements

- Break down larger functions into smaller, reusable components to improve maintainability and readability.

- Write unit tests for core functions and integration tests for the entire pipeline to ensure functionality and catch regressions.Set up continuous integration (CI) pipelines for automated testing on code commits.

- Use Docker to containerize the application, making it easy to deploy and scale across different environments.





