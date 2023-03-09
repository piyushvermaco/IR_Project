import os
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

# Set up YouTube API credentials
credentials, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/youtube.force-ssl"]
)
if not credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/youtube.force-ssl"]
    )

# Define function to retrieve transcript of a YouTube video
def get_video_transcript(video_id):
    youtube = build("youtube", "v3", credentials=credentials)

    try:
        # Retrieve the list of available captions for the video
        captions = youtube.captions().list(part="snippet", videoId=video_id).execute()

        # Look for captions that are marked as "auto-generated"
        auto_captions = [
            c for c in captions["items"] if c["snippet"]["isAutoSynced"]
        ]
        if not auto_captions:
            raise Exception("No auto-generated captions found for this video.")

        # Retrieve the transcript for the first auto-generated caption track
        caption = auto_captions[0]
        transcript = youtube.captions().download(id=caption["id"]).execute()

        # Convert the transcript to plain text format
        lines = transcript.split("\n")
        text = ""
        for line in lines:
            if line.strip() == "":
                continue
            line = line.replace("<[^>]*>", "")
            text += line.strip() + " "

        return text

    except HttpError as e:
        print(f"An error occurred: {e}")
        return None
