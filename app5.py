import streamlit as st
from transformers import pipeline, AutoTokenizer
from langdetect import detect
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import string
import re

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

@st.cache_resource
def load_llm_model():
    model_name = "unsloth/Llama-3.2-1B"  # Replace with your actual LLM model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
    return model

@st.cache_resource
def load_language_tool():
    return language_tool_python.LanguageTool('en-US')

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your actual model path

def keyword_matching(expected_answer, student_answer):
    key_tokens = word_tokenize(expected_answer.lower())
    student_tokens = word_tokenize(student_answer.lower())
    key_tokens = [word for word in key_tokens if word not in stopwords.words('english') and word not in string.punctuation]
    student_tokens = [word for word in student_tokens if word not in stopwords.words('english') and word not in string.punctuation]

    matches = set(key_tokens) & set(student_tokens)
    score = (len(matches) / len(key_tokens) * 10) if key_tokens else 0
    return score

def synonym_check(expected_answer, student_answer):
    key_tokens = word_tokenize(expected_answer.lower())
    student_tokens = word_tokenize(student_answer.lower())

    synonyms_found = 0
    for word in student_tokens:
        for syn in wordnet.synsets(word):
            if syn.name().split('.')[0] in key_tokens:
                synonyms_found += 1
                break

    score = (synonyms_found / len(key_tokens) * 10) if key_tokens else 0
    return score

def grammar_check(student_answer):
    grammar_tool = language_tool_python.LanguageTool('en-US')
    matches = grammar_tool.check(student_answer)
    num_errors = len(matches)
    if num_errors == 0:
        return 10
    elif num_errors < 3:
        return 8
    elif num_errors < 6:
        return 6
    else:
        return 4

def contextual_analysis(expected_answer, student_answer, sentence_model):
    key_embedding = sentence_model.encode(expected_answer, convert_to_tensor=True)
    student_embedding = sentence_model.encode(student_answer, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(key_embedding, student_embedding)
    score = cosine_similarity.item() * 10  # Scale the score to 0-10
    return score

def evaluate_answer(expected_answer, student_answer, sentence_model):
    # Calculate individual scores
    keyword_score = keyword_matching(expected_answer, student_answer)
    synonym_score = synonym_check(expected_answer, student_answer)
    grammar_score = grammar_check(student_answer)
    context_score = contextual_analysis(expected_answer, student_answer, sentence_model)

    # Adjusted weights
    final_score = (keyword_score * 0.3) + (synonym_score * 0.15) + (grammar_score * 0.10) + (context_score * 0.45)
    return final_score

def llm_score(llm_pipeline, question, student_answer, real_answer):
    prompt = """You are a teacher grading a quiz. You will be given a question, an expected answer, and a student's answer. Your task is to assign a score to the student out of 10 marks, following these grading guidelines that account for correctness, relevance, completeness, and the level of understanding shown.

### Grading Guidelines

- Yes/No Questions:
  - Assign a score of 10 if the student’s answer directly aligns with the intent of the expected answer, even if it’s brief (e.g., "yes" for "Do you own a pet?" with an expected answer of "Yes").
  - Assign a score of 7-9 if the answer is mostly correct but lacks confidence or has minor variations that still suggest correct intent.
  - Assign a score of 4-6 if the answer shows partial understanding, such as being vague or non-committal but suggesting partial alignment with the intent.
  - Give a score of 0-3 if the answer is incorrect, irrelevant, or shows no understanding of the question.

- Factual or Short-Answer Questions:
  - Assign a score of 10 if the student’s answer is factually correct, concise, and directly matches the expected answer.
  - Give a score between 7-9 if the answer contains mostly correct information but has minor inaccuracies or is slightly incomplete.
  - Assign a score between 4-6 if the answer includes some correct elements but lacks full accuracy or detail, indicating partial understanding.
  - Give a score of 1-3 if the answer is relevant but largely incorrect, with some minor connection to the correct answer.
  - Assign a score of 0 if the answer is incorrect, irrelevant, or entirely unrelated.

- Detailed or Open-Ended Questions:
  - Assign a score of 10 if the student’s answer covers all key points expected, provides additional relevant insights, or shows a deep understanding.
  - Give a score between 7-9 if the answer is mostly correct but may lack a few details, demonstrating a good understanding but not fully comprehensive.
  - Assign a score between 4-6 if the answer addresses some relevant aspects but is missing important details or contains notable inaccuracies, indicating partial understanding.
  - Assign a score between 1-3 if the answer shows limited relevance or understanding, with minimal correct information.
  - Assign a score of 0 if the answer is irrelevant, incorrect, or off-topic.

- Handling Negations and Contradictions:
  - If the student’s answer directly contradicts the expected answer by using words like "not," "never," or other negative phrasing that reverses the intended meaning, assign a score of 0, even if terminology or phrasing from the expected answer is used.
  - Assign a partial score only if the answer is partially correct but shows misunderstanding due to unnecessary negations or contradictory statements.

- Exceptional Cases:
  - For answers that are correct but significantly deviate in format from the expected answer (e.g., phrasing, synonyms, or alternative explanations), consider scoring between 7-10 based on relevance and understanding.
  - For answers that demonstrate creativity or unique perspectives in open-ended questions, reward additional points (up to 10) if they add value without deviating from the main intent of the question.
  - For unclear or ambiguous answers, make a best-effort assessment:
    - 5-7 if the response shows some alignment with the correct answer but is vague or poorly expressed.
    - 1-4 if there’s minimal alignment, suggesting some understanding but lacking clarity or focus.

### Question:
What is a large language model?

### Expected Answer:
A large language model (LLM) is a type of artificial intelligence (AI) program that can recognize and generate text, among other tasks. LLMs are trained on huge sets of data — hence the name large. LLMs are built on machine learning: specifically, a type of neural network called a transformer model.

### Student Answer:
An LLM is not an advanced artificial intelligence system that specializes in processing, understanding, and generating human-like text. These systems are not typically implemented as a deep neural network and are trained on massive amounts of text data.

### Score out of 10:
0

### Question:
{question}

### Expected Answer:
{expected_answer}

### Student Answer:
{student_answer}

### Score out of 10:
"""
    
    try:
        response = llm_pipeline(prompt, max_new_tokens=50)[0]['generated_text']
        match = re.search(r"Score\s*\(out of 10\):\s*(\d+(\.\d+)?)", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
    except Exception as e:
        print(f"Error in LLM scoring: {e}")
    return 0

def main():
    st.title("Answer Evaluation System")
    
    st.subheader("Inputs")
    question = st.text_area("Enter the Question:")
    real_answer = st.text_area("Enter the Real Answer:")
    student_answer = st.text_area("Enter the Student's Answer:")
    
    if st.button("Evaluate Answer"):
        llm_model = load_llm_model()
        sentence_model = load_sentence_transformer_model()  # Load the SentenceTransformer model
        language_tool = load_language_tool()
        
        st.subheader("Evaluation")
        
        # Sentence Transformer Score
        bert_eval_score = evaluate_answer(real_answer, student_answer, sentence_model)
        st.write(f"Sentence Transformer Evaluation Score: {bert_eval_score:.2f}")
        
        # LLM Score
        llm_eval_score = llm_score(llm_model, question, student_answer, real_answer)
        st.write(f"LLM Evaluation Score: {llm_eval_score:.2f}")
        
        # Weighted Final Score
        weighted_final_score = (bert_eval_score * 0.30) + (llm_eval_score * 0.70)
        st.write(f"Weighted Final Score: {weighted_final_score:.2f}")

if __name__ == "__main__":
    main()

