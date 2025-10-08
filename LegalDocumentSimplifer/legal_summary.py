# import torch
# from transformers import (
#     BartForConditionalGeneration,
#     BartTokenizer,
# )
# import streamlit as st
# from rouge_score import rouge_scorer
# import numpy as np
# import kagglehub



# model = BartForConditionalGeneration.from_pretrained('models/billsum')
# tokenizer = BartTokenizer.from_pretrained('models/billsum')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# model.eval()

# from summarygenerator import generate_summary
# st.title("Legal Text Summary Generator")
# text = st.text_area("Enter legal text:", height=100)
# if st.button("Generate Summary"):
#     if text.strip():  
#         summary = generate_summary(text)
#         st.subheader("Generated Summary")
#         st.write(summary)
#     else:
#         st.warning("Please enter some text to summarize.")
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
)
import streamlit as st
from rouge_score import rouge_scorer
import numpy as np
import kagglehub
import PyPDF2
import io

# Load model
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

def generate_summary(text, max_length=150, min_length=50):
    """Generate a summary for the given text using the BART model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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