import pandas as pd
import numpy as np
import re
import os
import random
import spacy
from spacy.matcher import Matcher
from transformers import pipeline
from transformers import AutoTokenizer, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Preprocessor():
    def __init__(self, yoe_patterns, degree_patterns):
        self.yoe_patterns = yoe_patterns
        self.degree_patterns = degree_patterns
        self.is_preprocessed()
        pass

    def is_preprocessed(self):
        path = 'job_preprocessed.csv'
        if os.path.exists(path):
            self.data = pd.read_csv(path)
        else:
            self.load_dataset()
            self.preprocess()
            self.extract_features()
            self.export_csv()

    def load_dataset(self):
        self.data = pd.read_csv('job_data.csv')
        self.data = self.data.drop_duplicates()
        self.original_data = self.data.copy()
    
    def preprocess(self):
        self.data['skills'] = self.data['skills'].astype(str)
        self.data['about'] = self.data['about'].astype(str)
        self.data.loc[~self.data['qualifications'].isna(), 'qualifications'] = self.data['qualifications'].dropna().astype(str)
        self.data['job_title'] = self.data['job_title'].apply(lambda x: x.lower())
        self.data['skills'] = self.data['skills'].apply(lambda x: x.lower())
        self.data['about'] = self.data['about'].apply(lambda x: x.lower())
        self.data['about'] = self.data['about'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
        self.data.loc[~self.data['qualifications'].isna(), 'qualifications'] = self.data['qualifications'].dropna().apply(lambda x: x.lower())
        self.data.loc[~self.data['qualifications'].isna(), 'qualifications'] = self.data['qualifications'].dropna().apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
    
    def get_degree(self, text):
        doc = self.nlp(text)
        matches = self.degree_matcher(doc)
        results = []
        for match_id, start, end in matches:
            span = doc[start:end]
            results.append(span.text)
        if len(results) == 0:
            return None
        return ','.join(results)

    def get_experience(self, text):
        doc = self.nlp(text)
        matches = self.yoe_matcher(doc)
        yoe_found = False
        results = []
        for match_id, start, end in matches:
            span = doc[start:end]
            results.append(span.text)
            yoe_found = True
        if len(results) == 0:   
            return None
        return ','.join(results)

    def get_degree_value(self, degree):
        if pd.isna(degree):
            return None
        degrees = degree.lower().split(",")
        min_degree = 4
        for d in degrees:
            if 'bachelor' in d or 'undergraduate' in d:
                min_degree = min(min_degree, 1)
            elif 'master' in d or 'mba' in d:
                min_degree = min(min_degree, 2)
            elif 'phd' in d or 'doctorate' in d:
                min_degree = min(min_degree, 3)
        return min_degree
    
    def get_min_experience(self, experience):
        if pd.isna(experience):
            return None
        exp = experience.split(",")
        min_exp = float('inf')
        for e in exp:
            match = re.search(r'\d+', e)
            if match:
                num = int(match.group())
                min_exp = min(min_exp, num)
        return min_exp

    def extract_features(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.yoe_matcher = Matcher(self.nlp.vocab)
        self.degree_matcher = Matcher(self.nlp.vocab)
        for pattern in yoe_patterns:
            self.yoe_matcher.add("YEARS_EXPERIENCE", [pattern])
        for pattern in degree_patterns:
            self.degree_matcher.add("DEGREE", [pattern])
        self.data['degree'] = self.data['qualifications'].apply(lambda x: self.get_degree(x) if pd.notna(x) else None)
        self.data['degree'] = self.data.apply(lambda x: self.get_degree(x['about']) if pd.isna(x['degree']) else x['degree'], axis=1)
        self.data['experience'] = self.data['about'].apply(lambda x: self.get_experience(x))
        self.data['experience'] = self.data.apply(lambda x: self.get_experience(x['about']) if pd.isna(x['experience']) else x['experience'], axis=1)
        self.data['degree_value'] = self.data['degree'].apply(lambda x: self.get_degree_value(x))
        self.data['min_experience'] = self.data['experience'].apply(lambda x: self.get_min_experience(x))
    
    def export_csv(self):
        self.data.to_csv('job_preprocessed.csv', index=False)

yoe_patterns = [
    # 2 years experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": {"IN": ["experience", "exp"]}}],
    # 2 years of experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "of"}, {"LOWER": {"IN": ["experience", "exp"]}}],
    # 2 years work experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": {"IN": ["work", "working", "professional"]}},{"LOWER": "experience"}],
    # 2 years of work experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "of"}, {"LOWER": {"IN": ["work", "working", "professional"]}},{"LOWER": "experience"}],
    # experience of 3 years
    [{"LOWER": {"IN": ["experience", "exp"]}}, {"LOWER": {"IN": ["of", ""]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}],
    # 2 years as a developer / 2 years in a similar role
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": {"IN": ["as", "in"]}}],
    # 1-2 years experience
    [{"LIKE_NUM": True}, {"IS_SPACE": True, "OP": "*"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}],
    # 2 years of relevant
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "of"}, {"LOWER": "relevant"}],
    # 2 years relevant 
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "relevant"}],
    # 2 years of hands on experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "of"}, {"LOWER": "hands"}, {"LOWER": "on"}, {"LOWER": "experience"}],
    # 2 years hands on experience
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "hands"}, {"LOWER": "on"}, {"LOWER": "experience"}],
]

degree_patterns = [
    [{"LOWER": {"IN": ["bachelor", "bachelors", "undergraduate"]}}],
    [{"LOWER": "master"}, {"LOWER": {"IN": ["'s", "s"]}}],
    [{"LOWER": "mba"}],
    [{"LOWER": {"IN": ["phd", "doctorate"]}}],
]

model = Preprocessor(yoe_patterns=yoe_patterns, degree_patterns=degree_patterns)