# 📜 Legal Text Summary Generator

## ✨ Overview
The **Legal Text Summary Generator** is a powerful web application designed to create concise summaries of legal documents. It leverages a fine-tuned **BART model (BillSum dataset)** for high-quality text summarization. Users can input text manually or upload a PDF document to extract and summarize its content.

🔹 **Built with:**
- 🖥 **Streamlit** - Interactive and intuitive UI
- 🤖 **Hugging Face Transformers** - Advanced text summarization

## 🚀 Features
✅ AI-powered summarization of legal documents using a **BART-based model**
✅ Extracts and processes text from **PDF files**
✅ **User-friendly interface** with simple input options
✅ Handles errors for scanned or protected PDFs
✅ Potential for **ROUGE metric evaluation**

## 🛠 Installation
To set up the application locally, follow these steps:

### 📥 1. Clone the Repository
```bash
git clone https://github.com/krishan098/legaldocumentsimplifier.git
cd legaldocumentsimplifier
```

### 📌 2. Install Dependencies
Ensure **Python 3.8+** is installed, then run:
```bash
pip install -r requirements.txt
```

### ▶ 3. Run the Application
```bash
streamlit run legal_summary.py
```

## 🎯 Usage Guide
1️⃣ **Enter text manually** and click **"Generate Summary"**
2️⃣ **Upload a PDF** and click **"Extract and Summarize"**
3️⃣ Expand the **"View Extracted Text"** section to see extracted content
4️⃣ Read the AI-generated summary in the **"Generated Summary"** section

## 📦 Dependencies
- 🔥 `torch` - Deep learning computations
- 📝 `transformers` - AI-driven text summarization
- 🎛 `streamlit` - Web-based UI framework
- 📊 `rouge_score` - Summary evaluation metric
- 📄 `PyPDF2` - PDF text extraction

## 🤖 Model Details
The summarization model used is **BART-large-CNN fine-tuned on BillSum**, a dataset specialized in summarizing legal and legislative documents.

## ⚠️ Limitations
🚧 Works best with well-formatted legal documents    
🚧 Summary accuracy depends on input text complexity  

## 🔮 Future Improvements
✨ Add **OCR support** for scanned PDFs  
✨ Enable **customizable summary length** options  
✨ Improve **text preprocessing** for better summarization  

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For questions or contributions, feel free to open an issue or reach out!

---
👤 **Author:** Drishti Bothra  
🌐 **GitHub:** drishtibothra  
📧 **Email:** drishti.bothra05@gmail.com

![image](https://github.com/user-attachments/assets/c4b233ee-e09b-47a0-8623-ad5a05caa39b)
![image](https://github.com/user-attachments/assets/6b41362d-82b9-4a5a-af07-9511e1949199)
![image](https://github.com/user-attachments/assets/eace5b30-b497-4c02-a8d1-01786bcd54c6)
![image](https://github.com/user-attachments/assets/e779df9b-723b-40c5-9227-0f8f950bbffc)


