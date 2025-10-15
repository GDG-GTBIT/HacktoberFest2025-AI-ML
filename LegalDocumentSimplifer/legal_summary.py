import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import streamlit as st
import PyPDF2
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load model
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("manueldeprada/FactCC")
    model = AutoModelForSequenceClassification.from_pretrained("manueldeprada/FactCC")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to(device)
    model.eval()
    return summarizer, model, tokenizer, device, similarity_model

summarizer, model, tokenizer, device, similarity_model = load_model()

def find_relevant_context(source_text, sentence, window=3):
    source_sentences = sent_tokenize(source_text)
    if not source_sentences:
        return source_text

    best_idx, best_score = 0, 0
    for i, s in enumerate(source_sentences):
        score = SequenceMatcher(None, sentence.lower(), s.lower()).ratio()
        if score > best_score:
            best_idx, best_score = i, score

    start = max(0, best_idx - window // 2)
    end = min(len(source_sentences), best_idx + window // 2 + 1)
    return " ".join(source_sentences[start:end])

def check_factual_consistency(source_text, summary_text, model, tokenizer, device):

    sentences = sent_tokenize(summary_text)
    labels = ["INCONSISTENT", "CONSISTENT"]

    results = []
    for sent in sentences:
        relevant_context = find_relevant_context(source_text, sent)

        inputs = tokenizer(
            relevant_context,
            sent,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            label = labels[pred]

        emb1 = similarity_model.encode(relevant_context, convert_to_tensor=True)
        emb2 = similarity_model.encode(sent, convert_to_tensor=True)
        sim = float(util.cos_sim(emb1, emb2))

        if label == "INCONSISTENT" and sim > 0.8:
            label = "LIKELY CONSISTENT"

        results.append((sent, label, round(sim, 3)))

    return results


def generate_summary(text, max_length=150, min_length=50):
    """Generate a summary for the given text using the BART model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    return summary[0]['summary_text']

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Streamlit UI
st.title("Legal Text Summary Generator")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Enter Text", "Upload PDF"])

with tab1:
    text_input = st.text_area("Enter legal text:", height=200)
    generate_button = st.button("Generate Summary", key="text_button")
    
    if generate_button and text_input.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(text_input)
            st.subheader("Generated Summary")
            st.write(summary)

            with st.spinner("Checking factual consistency..."):
                fact_results = check_factual_consistency(text_input, summary, model, tokenizer, device)

            st.subheader("Fact Consistency Results")
            for sent, label, points in fact_results:
                if label == "LIKELY CONSISTENT":
                    color = "🟡"
                elif label == "INCONSISTENT":
                    color = "🔴"
                else:
                    color = "🟢"
                st.markdown(f"{color} **{label}**: {sent}")
    elif generate_button:
        st.warning("Please enter some text to summarize.")

with tab2:
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write(file_details)
        
        # Extract text from PDF
        if st.button("Extract and Summarize", key="pdf_button"):
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                
                if extracted_text.strip():
                    # Show extracted text (collapsible)
                    with st.expander("View Extracted Text"):
                        st.text_area("Extracted Content:", value=extracted_text, height=200, disabled=True)
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(extracted_text)
                        st.subheader("Generated Summary")
                        st.write(summary)
                else:
                    st.error("Could not extract text from the PDF. The file might be scanned or protected.")

# Add some information about the app
with st.expander("About this app"):
    st.write("""
    This application uses the BART model fine-tuned on BillSum dataset to generate summaries of legal documents.
    Upload a PDF or paste text directly to get a concise summary.
    
    Note: For best results, use PDFs with proper text encoding rather than scanned documents.
    """)