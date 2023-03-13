import nltk
import numpy as np
import networkx as nx

function read_article(file_name):
    # read the file and split the article into sentences
    filedata = read_file(file_name)
    article = split_sentences(filedata)
    # split each sentence into words
    sentences = split_words(article)
    # remove the last sentence since it is usually incomplete
    remove_last_sentence(sentences)
    return sentences

function similarity_score(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
    # convert sentences to lowercase and remove stopwords
    sentence1 = lowercase_words(remove_stopwords(sentence1, stopwords))
    sentence2 = lowercase_words(remove_stopwords(sentence2, stopwords))
    # create a list of all words in the two sentences
    all_words = list(set(concatenate_lists(sentence1, sentence2)))
    # create a vector of word counts for each sentence
    vector1 = create_word_vector(sentence1, all_words)
    vector2 = create_word_vector(sentence2, all_words)
    # calculate the cosine distance between the two vectors
    return 1 - cosine_distance(vector1, vector2)

function build_similarity_matrix(sentences, stop_words):
    # create a similarity matrix for all pairs of sentences
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1, _ in enumerate(sentences):
        for idx2, _ in enumerate(sentences):
            if idx1 == idx2:
                continue
            # calculate the similarity score for each pair of sentences
            similarity_matrix[idx1][idx2] = similarity_score(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

function generate_summary(file_name, num_sentences=5):
    # download the stopwords if not already downloaded
    download_stopwords()
    # get the list of stopwords
    stop_words = get_stopwords()
    # initialize an empty list for the summary sentences
    summarize_text = []
    # read the article from file and split into sentences
    sentences = read_article(file_name)
    # build a similarity matrix for all pairs of sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    # create a graph from the similarity matrix
    sentence_similarity_graph = create_graph(sentence_similarity_matrix)
    # calculate the scores of each sentence using PageRank algorithm
    scores = calculate_pagerank(sentence_similarity_graph)
    # sort the sentences in descending order of scores
    ranked_sentences = sort_sentences_by_score(scores, sentences)
    # select the top 'num_sentences' sentences for the summary
    for i in range(num_sentences):
        summarize_text.append(join_words(ranked_sentences[i][1]))
    # join the selected sentences to form the summary
    summary = join_sentences(summarize_text)
    print("Summarized Text: \n", summary)
    return summary
