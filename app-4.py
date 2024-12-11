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
    model_name = "rohand8/Final"  # Replace with your actual LLM model path
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
    prompt = f"""

SCORING ARCHITECTURE:
I. ACCURACY DIMENSION (40 Points)
- Binary verification against key answer
- Semantic similarity analysis
- Contextual alignment check
- Zero-tolerance for fundamental errors

SCORING SUB-COMPONENTS:
A. Exact Match Verification (15 pts)
- Literal keyword matching
- Precise terminological accuracy
- Complete answer coverage

B. Semantic Understanding (15 pts)
- Conceptual comprehension depth
- Contextual reasoning
- Nuanced interpretation potential

C. Critical Knowledge Mapping (10 pts)
- Core concept identification
- Domain-specific knowledge integration
- Systemic understanding evaluation

II. COMPREHENSION DIMENSION (30 Points)
- Reasoning quality assessment
- Explanation coherence
- Logical connection demonstration
- Complexity of understanding

III. TECHNICAL PRECISION (20 Points)
- Terminology accuracy
- Domain-specific language use
- Technical concept representation

IV. STRUCTURAL INTEGRITY (10 Points)
- Answer organization
- Clarity of communication
- Logical flow

ADAPTIVE SCORING MECHANISM:
- Fundamentally incorrect: 0-2/10
- Partially correct: 3-5/10
- Substantially accurate: 6-8/10
- Exceptional understanding: 9-10/10

OUTPUT REQUIREMENTS:
- Detailed score decomposition
- Specific error/gap identification
- Constructive improvement recommendations
- Authoritative source references

PROCESSING ALGORITHM:
1. Extract key concepts from key answer
2. Compare against student response
3. Apply multi-dimensional scoring matrix
4. Generate comprehensive evaluation report


### Question:
{question}

### Expected Answer:
{real_answer}

### Student Answer:
{student_answer}

### Score (out of 10):"""
    
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
    question = st.text_area("Question:")
    real_answer = st.text_area("Key Answer:")
    student_answer = st.text_area("Student's Answer:")
    
    if st.button("Evaluate Answer"):
        llm_model = load_llm_model()
        sentence_model = load_sentence_transformer_model()  # Load the SentenceTransformer model
        language_tool = load_language_tool()
        
        st.subheader("Evaluation")
        
        # Sentence Transformer Score
        bert_eval_score = evaluate_answer(real_answer, student_answer, sentence_model)
        st.write(f"MiniLM L6 V2 Evaluation Score: {bert_eval_score:.2f}")
        
        # LLM Score
        llm_eval_score = llm_score(llm_model, question, student_answer, real_answer)
        st.write(f"LLaMA 3 8b Evaluation Score: {llm_eval_score:.2f}")
        
        # Weighted Final Score
        weighted_final_score = (bert_eval_score * 0.30) + (llm_eval_score * 0.70)
        st.write(f"Final Score: {weighted_final_score:.2f}")

if __name__ == "__main__":
    main()
