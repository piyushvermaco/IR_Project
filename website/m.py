from flask import Blueprint, render_template, request
from flask import Flask
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict
from transformers import pipeline
import transformers
import spacy

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def base():
    if request.method == 'POST':
        data = request.form

        # Set the YouTube video URL
        video_url = data.get('search_string') 

        # Extract the video ID from the URL
        if video_url is not None:
            video_id = video_url.split("=")[1]
        else:
            return render_template("base.html")


        # Get the transcript using the video ID
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Convert the list of transcript dictionaries to a string
        transcript_str = ""
        for text in transcript_list:
            transcript_str += text['text'] + " "

        clean_text = transcript_str.replace("'", "")
        
        # Load the pre-trained model 
        nlp = spacy.load("en_core_web_sm")
        model_name = "distilbert-base-uncased"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)

        def extract_info(text, nlp, tokenizer, model):
            # Split the input text into chunks of 10000 words
            chunk_size = 10000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            # Initialize empty lists for the output entities
            dates = []
            times = []
            locations = []
            keypoints = []

            # Process each chunk of text
            for chunk in chunks:
                # Tokenize the chunk
                inputs = tokenizer(chunk, return_tensors="pt")

                # Make predictions
                outputs = model(**inputs)

                # Get the predicted labels
                labels = outputs.logits.argmax(dim=2)

                # Convert labels back to strings
                label_strings = []
                for label in labels[0]:
                    label_strings.append(model.config.id2label[label.item()])

                # Use SpaCy to extract named entities and keypoints from the chunk
                doc = nlp(chunk)
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        dates.append(ent.text)
                    elif ent.label_ == "TIME":
                        times.append(ent.text)
                    elif ent.label_ == "GPE" or ent.label_ == "LOC":
                        locations.append(ent.text)
                for sentence in doc.sents:
                    sentence_doc = nlp(sentence.text)
                    for chunk in sentence_doc.noun_chunks:
                        if "NN" in chunk.root.tag_:
                            keypoints.append(chunk.text)

            # Extract speaker and topic information from the entire text
            doc = nlp(text)
            speaker = ""
            topic = ""
            for i in range(len(label_strings)):
                if label_strings[i] == "B-PER":
                    speaker = doc[i].text
                elif label_strings[i] == "B-TOPIC":
                    topic = doc[i].text
                elif label_strings[i] == "I-TOPIC":
                    topic += " " + doc[i].text

            # Use Hugging Face to extract sentiment from the entire text
            sentiment_analyzer = transformers.pipeline("sentiment-analysis")
            sentiment = sentiment_analyzer(text)[0]["label"]

            # Return the results as a dictionary
            return {
                "dates": dates,
                "times": times,
                "locations": locations,
                "speaker": speaker,
                "topic": topic,
                "keypoints": keypoints,
                "sentiment": sentiment
            } 
        
        summarizer = pipeline("summarization")

        def generate_notes(text, summarizer, chunk_size=1000):
            # Set summarization parameters
            max_length = 100
            min_length = 30
            do_sample = False

            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            # Generate notes
            notes = []
            for chunk in chunks:
                chunk_notes = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
                notes.append(chunk_notes[0]["summary_text"].strip())

            # Combine notes into single summary
            summary = " ".join(notes)

            return summary


        
        short_summary = generate_notes(clean_text,summarizer, chunk_size=1000)
        notes = extract_info(clean_text, nlp, tokenizer, model)
        
        
        return render_template("output.html",notes =notes,  short_summary = short_summary)
    else:
        return render_template('base.html')

