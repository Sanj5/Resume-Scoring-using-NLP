import streamlit as st
import PyPDF2  
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
   
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)


def match_job_description(job_description, resume_text):
   
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_resume_text = preprocess_text(resume_text)
    
    vectorizer = TfidfVectorizer()
    
    job_description_vector = vectorizer.fit_transform([preprocessed_job_description])
    
    resume_vector = vectorizer.transform([preprocessed_resume_text])
   
    similarity_score = cosine_similarity(job_description_vector, resume_vector)[0][0]
    return similarity_score


st.title("Resume Matcher")


job_description = st.text_area("Enter the job description")

st.write("Upload your resume to match it with the job description")


uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])

if uploaded_file is not None:
   
    resume_text = ""
    if uploaded_file.name.endswith('.txt'):
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith('.pdf'):
        
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page_num in range(len(pdf_reader.pages)):
            resume_text += pdf_reader.pages[page_num].extract_text()
    
    
    similarity_score = match_job_description(job_description, resume_text)
    
    
    st.write("Similarity Score:", similarity_score)
