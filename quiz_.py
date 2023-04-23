import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load('en_core_web_sm')


def generate_questions(text):
    
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
    
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]
    word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    
    
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words if word.lower() not in entities][:5]
    
    
    questions = {}
    for i, keyword in enumerate(keywords):
        for sent in doc.sents:
            if keyword in sent.text:
                question = sent.text.replace(keyword, '__________')
                questions[i+1] = {'question': question, 'answer': keyword}
                break
                
    return questions


text = "insert text here"


questions = generate_questions(text)

# Print the questions
for q_number, q_info in questions.items():
    print(f"Question {q_number}: {q_info['question']} (Answer: {q_info['answer']})")
