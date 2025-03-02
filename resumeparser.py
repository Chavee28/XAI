import os
from dotenv import load_dotenv
import PyPDF2
from groq import Groq
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer
import shap
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def parse_resume(text):
    """Parse resume text using Groq API"""
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    prompt = f"""Extract the following information from this resume. Format as JSON:
    - Full Name
    - Email
    - Skills (as a list)
    - Work Experience (as a list of dictionaries with company, position, and duration)
    - GitHub Profile (if present)
    - LinkedIn Profile (if present)

    Resume text:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Groq's model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return None

class ResumeMatcherXAI:
    def __init__(self):
        # Load the model and tokenizer
        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def get_embedding(self, text):
        """Get embeddings using the transformer model"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]

    def prepare_text(self, parsed_resume):
        """Convert parsed resume JSON to text"""
        try:
            if isinstance(parsed_resume, str):
                resume_data = json.loads(parsed_resume)
            else:
                resume_data = parsed_resume

            # Convert keys to match the expected format
            standardized_data = {}
            for key, value in resume_data.items():
                # Convert keys to the standard format
                if key.lower() == 'full name':
                    standardized_data['Full Name'] = value
                elif key.lower() == 'email':
                    standardized_data['Email'] = value
                elif key.lower() == 'skills':
                    standardized_data['Skills'] = value
                elif key.lower() == 'work experience':
                    standardized_data['Work Experience'] = value
                elif key.lower() == 'github profile':
                    standardized_data['GitHub Profile'] = value
                elif key.lower() == 'linkedin profile':
                    standardized_data['LinkedIn Profile'] = value
                else:
                    standardized_data[key] = value

            text_parts = []
            
            # Use standardized keys
            if 'Full Name' in standardized_data:
                text_parts.append(f"Name: {standardized_data['Full Name']}")
            
            if 'Email' in standardized_data:
                text_parts.append(f"Email: {standardized_data['Email']}")
            
            if 'Skills' in standardized_data and isinstance(standardized_data['Skills'], list):
                text_parts.append("Skills: " + ", ".join(standardized_data['Skills']))
            
            if 'Work Experience' in standardized_data and isinstance(standardized_data['Work Experience'], list):
                experience = []
                for exp in standardized_data['Work Experience']:
                    exp_text = f"{exp.get('position', '')} at {exp.get('company', '')}"
                    if exp.get('duration'):
                        exp_text += f" for {exp.get('duration')}"
                    experience.append(exp_text)
                if experience:
                    text_parts.append("Experience: " + "; ".join(experience))
            
            if 'GitHub Profile' in standardized_data:
                text_parts.append(f"GitHub: {standardized_data['GitHub Profile']}")
            
            if 'LinkedIn Profile' in standardized_data and standardized_data['LinkedIn Profile']:
                text_parts.append(f"LinkedIn: {standardized_data['LinkedIn Profile']}")
            
            combined_text = " ".join(text_parts)
            print("Prepared Text:", combined_text)  # Debug print
            return combined_text if combined_text.strip() else "No relevant information found"
        
        except Exception as e:
            print(f"Error in prepare_text: {e}")
            return str(parsed_resume)

    def get_similarity(self, resume_embedding, job_embedding):
        """Calculate cosine similarity between embeddings"""
        return cosine_similarity(
            resume_embedding.reshape(1, -1), 
            job_embedding.reshape(1, -1)
        )[0][0]

    def get_lime_explanation(self, resume_text, job_description):
        """Generate LIME explanation for the match"""
        explainer = LimeTextExplainer(class_names=['Not Match', 'Match'])
        
        def predict_proba(texts):
            probs = []
            for text in texts:
                emb1 = self.get_embedding(text)
                emb2 = self.get_embedding(job_description)
                sim = self.get_similarity(emb1, emb2)
                probs.append([1-sim, sim])
            return np.array(probs)
        
        exp = explainer.explain_instance(
            resume_text,
            predict_proba,
            num_features=10,
            num_samples=100
        )
        
        return [{'feature': feat, 'importance': round(imp, 4)} 
                for feat, imp in exp.as_list()]

    def match_resume(self, resume_text, job_description):
        """Match resume against job description with improved analysis"""
        try:
            if not resume_text or not job_description:
                raise ValueError("Resume text or job description is empty")

            # Get embeddings
            resume_embedding = self.get_embedding(resume_text)
            job_embedding = self.get_embedding(job_description)
            
            # Calculate similarity score
            similarity = self.get_similarity(resume_embedding, job_embedding)
            score = round(similarity * 100, 2)
            
            # Improved LIME explanation focusing on meaningful terms
            def get_meaningful_terms(text):
                # Split text into words
                words = text.lower().split()
                # Filter out common words, dates, and prepositions
                stop_words = {'at', 'for', 'in', 'on', 'to', 'the', 'and', 'or', 'of', 'with', 'by',
                             'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                             'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                             'october', 'november', 'december'}
                return [w for w in words if w not in stop_words and len(w) > 2]

            # Get meaningful terms from both texts
            resume_terms = set(get_meaningful_terms(resume_text))
            job_terms = set(get_meaningful_terms(job_description))

            # Calculate term importance
            term_importance = []
            
            # Technical skills importance (higher weight)
            technical_skills = resume_terms.intersection(job_terms)
            for skill in technical_skills:
                if any(tech in skill.lower() for tech in ['python', 'java', 'c++', 'javascript', 'aws', 'cloud', 'ml', 'ai']):
                    term_importance.append({
                        'feature': skill,
                        'importance': round(float(similarity * 1.5), 4),
                        'category': 'Technical Skill'
                    })

            # Experience and role matches
            experience_terms = [term for term in resume_terms.intersection(job_terms)
                              if any(role in term.lower() for role in ['engineer', 'developer', 'analyst', 'manager', 'intern'])]
            for term in experience_terms:
                term_importance.append({
                    'feature': term,
                    'importance': round(float(similarity * 1.2), 4),
                    'category': 'Role'
                })

            # Domain knowledge
            domain_terms = resume_terms.intersection(job_terms) - set(technical_skills) - set(experience_terms)
            for term in domain_terms:
                term_importance.append({
                    'feature': term,
                    'importance': round(float(similarity), 4),
                    'category': 'Domain Knowledge'
                })

            # Missing important terms from job description
            missing_terms = [term for term in job_terms 
                            if term not in resume_terms and 
                            any(tech in term.lower() for tech in ['python', 'java', 'c++', 'javascript', 'aws', 'cloud', 'ml', 'ai'])]
            for term in missing_terms[:3]:  # Show top 3 missing terms
                term_importance.append({
                    'feature': term,
                    'importance': round(float(-0.5), 4),
                    'category': 'Missing Skill'
                })

            # Sort by absolute importance and take top 10
            term_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
            term_importance = term_importance[:10]

            return {
                'score': score,
                'explanations': term_importance
            }
            
        except Exception as e:
            print(f"Error in match_resume: {e}")
            raise Exception(f"Matching error: {e}")

# Initialize the matcher
matcher = ResumeMatcherXAI()

# Functions to be used by the Flask app
def prepare_text_for_matching(parsed_resume):
    return matcher.prepare_text(parsed_resume)

def match_resume(resume_text, job_description):
    return matcher.match_resume(resume_text, job_description)

def main():
    # Replace with your PDF path
    pdf_path = "path_to_your_resume.pdf"
    
    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_path)
    if resume_text:
        # Parse resume
        parsed_data = parse_resume(resume_text)
        print(parsed_data)

if __name__ == "__main__":
    main() 