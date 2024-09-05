import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class JobRecommender:
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), 'job_preprocessed.csv')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        self.data = pd.read_csv(file_path)
        self.data = self.data.head(250)
        self.load_model()

    def load_model(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def calculate_title_similarity(self, job_title, user_titles):
        job_title = job_title.lower()
        user_titles = user_titles.lower().split(",")
        concat = [job_title] + user_titles
        embeddings = self.model.encode(concat)
        job_title_embedding = embeddings[0]
        user_embeddings = embeddings[1:]
        similarities = cosine_similarity([job_title_embedding], user_embeddings)[0]
        return max(similarities) * 100

    def calculate_skill_similarity(self, job_skills, user_skills):
        if pd.isna(job_skills) or job_skills.strip() == "":
            return 0, None
        if pd.isna(user_skills) or user_skills.strip() == "":
            return 0, None

        job_skills = job_skills.lower().split(",")
        user_skills = user_skills.lower().split(",")
        
        job_skills = [skill.strip() for skill in job_skills]
        user_skills = [skill.strip() for skill in user_skills]
        
        concat = job_skills + user_skills
        embeddings = self.model.encode(concat)
        job_embeddings = embeddings[:len(job_skills)]
        user_embeddings = embeddings[len(job_skills):]
        
        matches = []
        score = 0

        for i in range(len(job_skills)):
            if i < len(user_skills) and job_skills[i] == user_skills[i]:
                score += 1
                matches.append({"matched_skill": job_skills[i], "user_skill": user_skills[i], "similarity": 100})
            else:
                similarities = cosine_similarity([job_embeddings[i]], user_embeddings)[0]
                max_similarity = max(similarities)
                max_index = np.argmax(similarities)
                
                if max_similarity > 0.4:
                    matched_user_skill = user_skills[max_index] if max_index < len(user_skills) else None
                    matches.append({
                        "matched_skill": job_skills[i], 
                        "user_skill": matched_user_skill, 
                        "similarity": round(max_similarity * 100, 3)
                    })
                    score += max_similarity
        
        score = score / len(job_skills) * 100

        if len(matches) == 0:
            matches = None
        else:
            matches = pd.DataFrame(matches).to_json(orient='records')

        return score, matches


    def get_degree_value(self, degree):
        if pd.isna(degree):
            return None
        degrees = degree.lower().split(",")
        min_degree = 4
        for d in degrees:
            if 'no degree' in d:
                min_degree = min(min_degree, 0)
            elif 'bachelor' in d or 'undergraduate' in d:
                min_degree = min(min_degree, 1)
            elif 'master' in d or 'mba' in d:
                min_degree = min(min_degree, 2)
            elif 'phd' in d or 'doctorate' in d:
                min_degree = min(min_degree, 3)
        return min_degree

    def calculate_degree_similarity(self, job_degree, user_degree):
        user_degree = self.get_degree_value(user_degree)
        job_degree = self.get_degree_value(job_degree)
        if pd.isna(job_degree):
            return -1
        if pd.isna(user_degree):
            return 0
        if user_degree >= job_degree:
            return 100
        return 0

    def calculate_experience_similarity(self, job_experience, user_experience):
        if pd.isna(job_experience):
            return -1
        if pd.isna(user_experience):
            return 0
        if job_experience <= user_experience:
            return 100
        return 0

    def calculate_overall_score(self, title_score, skill_score, degree_score, experience_score):
        if experience_score == -1 and degree_score == -1:
            return (skill_score * 0.5) + (experience_score * 0.5)
        elif experience_score == -1:
            return (skill_score * 0.45) + (experience_score * 0.45) + (degree_score * 0.1)
        elif degree_score == -1:
            return (title_score * 0.40) + (skill_score * 0.40) + (experience_score * 0.2)
        return (title_score * 0.40) + (skill_score * 0.40) + (degree_score * 0.05) + (experience_score * 0.15)

    def recommend_jobs(self, user_data):
        self.batch_size = 5
        self.job_ids = []
        self.match_ids = []
        self.title_scores = []
        self.skill_scores = []
        self.skill_matches = []
        self.degree_scores = []
        self.experience_scores = []
        self.overall_scores = []
        
        for i in range(len(self.data)):
            job = self.data.iloc[i]
            self.match_ids.append(i)
            print(f"Processing job {i + 1}/{len(self.data)}")
            self.job_ids.append(job['job_id'])
            
            title_score = self.calculate_title_similarity(job['job_title'], user_data['job_title'].iloc[0])
            self.title_scores.append(title_score)
            
            skill_score, skill_matches = self.calculate_skill_similarity(job['skills'], user_data['skills'].iloc[0])
            self.skill_scores.append(skill_score)
            
            self.skill_matches.append(skill_matches if skill_matches is not None else '[]')
            
            degree_score = self.calculate_degree_similarity(job['degree'], user_data['degree'].iloc[0])
            self.degree_scores.append(degree_score)
            
            experience_score = self.calculate_experience_similarity(job['min_experience'], user_data['years_of_experience'].iloc[0])
            self.experience_scores.append(experience_score)
            
            overall_score = self.calculate_overall_score(title_score, skill_score, degree_score, experience_score)
            self.overall_scores.append(overall_score)

        self.result = pd.DataFrame({
            "id": self.job_ids,
            "title_score": self.title_scores,
            "skill_score": self.skill_scores,
            "degree_score": self.degree_scores,
            "experience_score": self.experience_scores,
            "similarity": self.overall_scores
        })

        self.result = self.result.sort_values(by='similarity', ascending=False)
        self.skill_matches = pd.DataFrame({"job_id": self.job_ids, "skill_matches": self.skill_matches})
        self.skill_matches = self.skill_matches.set_index("job_id").reindex(self.result['id']).reset_index()
        self.skill_matches = self.skill_matches.dropna(subset=['skill_matches'])
        
        return self.result, self.skill_matches


if __name__ == "__main__":
    user_data = {
        "user_id": 1,
        "job_title": "Data Scientist,Data Analyst,Data Engineer,Machine Learning Engineer",
        "skills": "Python,Machine Learning,Deep Learning,Data Science,Data Analysis,Data Engineering,SQL,Tensorflow,Pytorch,Scikit-learn,Natural Language Processing,R",
        "degree": "Bachelor's Degree",
        "years_of_experience": 1
    }
    user_data_df = pd.DataFrame(user_data, index=[0])
    model = JobRecommender()
    result, skill_matches = model.recommend_jobs(user_data_df)
    print(result.to_json(orient='records'))
    print(skill_matches.to_json(orient='records'))
