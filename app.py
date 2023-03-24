# Import the necessary library
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import CountVectorizer
import re
from typing import List, Dict

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Set the YouTube video URL
video_url = "https://www.youtube.com/watch?v=6VTFx0U7YAo&list=PLSQl0a2vh4HB0jC9LI1SktmkZGzd_oz59"

# Extract the video ID from the URL
video_id = video_url.split("=")[1]

# Getting the transcript using the YouTube video API
transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

# Converting the list of transcript dictionaries to a string

trnscrpt_str = ""
for text in transcript_list:
    trnscrpt_str += text['text'] + " "

clean_text = trnscrpt_str.replace("'", "")


# This is the function used for making the summary 

def summarize(text, summary_length=7):
    # Tokenize the text into sentences
    sntnc = text.split('.')
    
    # Remove any leading/trailing white space from each sentence
    sntnc = [s.strip() for s in sntnc]
    
    # Remove any empty sentences
    sntnc = [s for s in sntnc if len(s) > 0]
    
    # Create a feature matrix using a bag-of-words model
    vectorizer = CountVectorizer(stop_words='english')
    sntnc_mtrx = vectorizer.fit_transform(sntnc).toarray()
    
    # Apply Latent Semantic Analysis (LSA) to reduce the dimensionality of the feature matrix
    lsa = TruncatedSVD(n_components=min(len(sntnc)-1, 100), algorithm='randomized', n_iter=10, random_state=42)
    sntnc_mtrx = lsa.fit_transform(sntnc_mtrx)
    
    # Calculate the sentence scores based on their cosine similarity to the document vectors
    doc_vector = sntnc_mtrx.mean(axis=0)
    scores = {}
    for i in range(len(sntnc)):
        sent_vector = sntnc_mtrx[i]
        score = np.dot(sent_vector, doc_vector) / (np.linalg.norm(sent_vector) * np.linalg.norm(doc_vector))
        scores[sntnc[i]] = score
    
    # Getting the top n sentences with the highest scores, where n is the specified summary length that we want to get.
    topnsntnc = sorted(scores, key=scores.get, reverse=True)[:summary_length]
    
    # Combine the top n sntnc into a summary paragraph
    summary = '. '.join(topnsntnc)
    
    return summary


def make_notes(text: str) -> Dict[str, str]:
    
    # Define regular expressions to extract relevant information
    date_regex = r"\d{1,2}/\d{1,2}/\d{2,4}"  # Date in format "mm/dd/yyyy" or "m/d/yyyy"
    time_regex = r"\d{1,2}:\d{2}"  # Time in format "hh:mm"
    location_regex = r"(?<=at\s)[A-Za-z\s]+(?=\.)"  # Location mentioned as "at <location>."
    speaker_regex = r"[A-Z][a-z]+\s[A-Z][a-z]+"  # Speaker name in format "First Last"
    topic_regex = r"(?<=\n)[A-Za-z\s]+(?=:)"  # Topic mentioned as "<topic>:"
    sentiment_regex = r"positive|negative|neutral"  # Sentiment mentioned as "positive", "negative", or "neutral"
    key_points_regex = r"(?<=\n- )[A-Za-z\s]+(?=\.)"  # Key points mentioned as "- <key point>."
    
    # Search for matches using the regular expressions
    date_match = re.search(date_regex, text)
    time_match = re.search(time_regex, text)
    location_match = re.search(location_regex, text)
    speaker_match = re.search(speaker_regex, text)
    topic_match = re.search(topic_regex, text)
    sentiment_match = re.search(sentiment_regex, text)
    key_points_matches = re.findall(key_points_regex, text)
    
    # Create a dictionary of the extracted information
    notes = {
        "date": date_match.group(0) if date_match else None,
        "time": time_match.group(0) if time_match else None,
        "location": location_match.group(0) if location_match else None,
        "speaker": speaker_match.group(0) if speaker_match else None,
        "topic": topic_match.group(0) if topic_match else None,
        "sentiment": sentiment_match.group(0) if sentiment_match else None,
        "key_points": key_points_matches if key_points_matches else None
    }
    
    # Return the dictionary
    return notes


summary = summarize(clean_text)
print("Summary : ")
for i, sentence in enumerate(summary.split('. ')):
    print(f"{i+1}. {sentence}")
    
print("\n")

notes = make_notes(clean_text)

def print_notes(notes):
    print("Important events and keywords:")
    print("-----------------------------")
    if "action" in notes and notes["action"]:
        print(f"- Action: {notes['action']}")
    if "location" in notes and notes["location"]:
        print(f"- Location: {notes['location']}")
    if "date" in notes and notes["date"]:
        print(f"- Date: {notes['date']}")
    if "time" in notes and notes["time"]:
        print(f"- Time: {notes['time']}")
    if not any(notes.values()):
        print("No important events or keywords found.")


print_notes(notes)
