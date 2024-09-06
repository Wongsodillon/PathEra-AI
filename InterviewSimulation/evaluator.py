import mysql.connector
import pandas as pd
import torch
from rake_nltk import Rake
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Evaluator:
    def __init__(self, db_config, model_name='sentence-transformers/all-MiniLM-L12-v2'):
        self.connection = mysql.connector.connect(**db_config)
        self.model = SentenceTransformer(model_name)
        self.rake = Rake()
    
    def check_question_exists(self, question_id):
        query = f"SELECT 1 FROM questions WHERE id = {question_id} LIMIT 1"
        df = pd.read_sql(query, self.connection)
        return not df.empty

    def get_topic(self, question_id):
        query = f"SELECT topic FROM questions WHERE id = {question_id}"
        df = pd.read_sql(query, self.connection)
        topic = df['topic'][0]
        topic = topic.split(",")
        topic = [self.clean_text(t) for t in topic]  
        return topic

    def clean_text(self, text):
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  
        text = text.lower() 
        return text

    def extract_keywords(self, text, min_length=2, max_length=6):
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()
        keywords = [keyword for keyword in keywords if len(keyword.split()) >= min_length and len(keyword.split()) <= max_length]
        return keywords
    
    def get_keyword_embeddings(self, keywords):
        if not keywords: 
            return torch.zeros((1, self.model.get_sentence_embedding_dimension()))  
        return self.model.encode(keywords, convert_to_tensor=True)

    def aggregate_embeddings(self, embeddings):
        if embeddings.size(0) == 0:
            return torch.zeros(self.model.get_sentence_embedding_dimension())
        return embeddings.mean(dim=0)
    
    def get_reference_answers(self, question_id):
        answers = pd.read_sql(f"SELECT answer FROM answer_key WHERE question_id = {question_id}", self.connection)
        return answers['answer'].tolist()

    def preprocess_reference_answers(self, reference_answers, topic):
        def clean_and_replace(text):
            cleaned_text = self.clean_text(text)
            for t in topic:
                cleaned_text = cleaned_text.replace(t, "")
            return cleaned_text
        
        return [clean_and_replace(answer) for answer in reference_answers]

    def preprocess_user_answer(self, user_answer, topic):
        user_answer = self.clean_text(user_answer)
        for t in topic:
            user_answer = user_answer.replace(t, "") 
        return user_answer

    def calculate_keyword_similarity(self, user_answer, reference_answers):
        user_keywords = self.extract_keywords(user_answer)
        reference_keywords_list = [self.extract_keywords(answer) for answer in reference_answers]

        user_embeddings = self.get_keyword_embeddings(user_keywords)
        reference_embeddings = [self.get_keyword_embeddings(keywords) for keywords in reference_keywords_list]

        user_embedding = self.aggregate_embeddings(user_embeddings)
        reference_embeddings = [self.aggregate_embeddings(ref_embeddings) for ref_embeddings in reference_embeddings]

        similarities = []
        for ref_embedding in reference_embeddings:
            similarity = cosine_similarity(user_embedding.unsqueeze(0).cpu().numpy(), ref_embedding.unsqueeze(0).cpu().numpy())
            similarities.append(similarity[0][0])
        return max(similarities)

    def calculate_similarity(self, user_answer, reference_answers):
        user_embedding = self.model.encode(user_answer)
        reference_embeddings = self.model.encode(reference_answers)

        similarities = cosine_similarity([user_embedding], reference_embeddings)
        return max(similarities[0])

    def evaluate(self, question_id, user_answer):
        reference_answers = self.get_reference_answers(question_id)
        topic = self.get_topic(question_id)
        reference_answers = self.preprocess_reference_answers(reference_answers, topic)
        user_answer = self.preprocess_user_answer(user_answer, topic)
        keyword_similarity = self.calculate_keyword_similarity(user_answer, reference_answers)
        similarity = self.calculate_similarity(user_answer, reference_answers)
        print(f"Similarity: {similarity}, Keyword Similarity: {keyword_similarity}")
        final_score = (similarity * 0.4) + (keyword_similarity * 0.6)
        return round(final_score * 100, 2)

    def close_connection(self):
        self.connection.close()

# db_config = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': '',
#     'database': 'pathera',
#     'port': 3306
# }

# evaluator = Evaluator(db_config=db_config)
# question_id = 39
# user_answer = "is a dimensionality reduction technique that nonlinearly maps high-dimensional data to a lower-dimensional space, with a focus on preserving the local relationships between points."
# similarity_score = evaluator.evaluate(question_id, user_answer)
# print(f"Similarity Score: {similarity_score}")
