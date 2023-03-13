# Import the necessary library
from youtube_transcript_api import YouTubeTranscriptApi

# Set the YouTube video URL
video_url = "https://www.youtube.com/watch?v=0lSTXtwPuOU&list=PLSQl0a2vh4HBxoP1tZaejDjVn2Ysf_WDj"

# Extract the video ID from the URL
video_id = video_url.split("=")[1]

# Get the transcript using the video ID
transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

# Convert the list of transcript dictionaries to a string
transcript_str = ""
for text in transcript_list:
    transcript_str += text['text'] + " "

# Print the transcript
print(transcript_str)
