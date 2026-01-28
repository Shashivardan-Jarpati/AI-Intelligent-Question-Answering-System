import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords only (safe for Python 3.13)
nltk.download('stopwords')

# -------------------------------
# Text preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# -------------------------------
# Load knowledge base
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(BASE_DIR, "knowledge_base.txt")

with open(kb_path, "r", encoding="utf-8") as file:
    knowledge = file.readlines()

processed_knowledge = []
for line in knowledge:
    processed_knowledge.append(preprocess(line))

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
knowledge_vectors = vectorizer.fit_transform(processed_knowledge)

# -------------------------------
# Abbreviations
# -------------------------------
abbreviations = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "nlp": "natural language processing"
}

print("\nAI Intelligent Question Answering System")
print("Type 'exit' to quit")

# -------------------------------
# Chat loop
# -------------------------------
while True:
    user_query = input("\nAsk a question: ").strip()

    if user_query.lower() == "exit":
        print("Thank you!")
        break

    query = user_query.lower()

    for short, full in abbreviations.items():
        query = query.replace(short, full)

    processed_query = preprocess(query)
    query_vector = vectorizer.transform([processed_query])

    similarity = cosine_similarity(query_vector, knowledge_vectors)

    if similarity.max() < 0.2:
        print("AI Answer: Sorry, I don't have information on that.")
        continue

    best_match = similarity.argmax()
    print("AI Answer:", knowledge[best_match].strip())
