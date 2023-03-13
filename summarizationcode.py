import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.read().replace('\n', '')
    article = filedata.split(". ")
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in article]
    sentences.pop() 
    return sentences

def similarity_score(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sentence1 = [word.lower() for word in sentence1 if word.lower() not in stopwords]
    sentence2 = [word.lower() for word in sentence2 if word.lower() not in stopwords]
    all_words = list(set(sentence1 + sentence2))
    vector1 = [sentence1.count(word) for word in all_words]
    vector2 = [sentence2.count(word) for word in all_words]
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1, _ in enumerate(sentences):
        for idx2, _ in enumerate(sentences):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = similarity_score(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, num_sentences=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences =  read_article(file_name)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((score, sentence) for score, sentence in zip(scores, sentences)), reverse=True)
    for i in range(num_sentences):
        summarize_text.append(" ".join(ranked_sentences[i][1]))
    summary = ". ".join(summarize_text)
    print("Summarized Text: \n", summary)
    return summary

generate_summary("abcd.txt", 2)