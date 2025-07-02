# AI/ML Multi-Project Hub

This project combines five different AI/ML applications into a single FastAPI web application with a simple, dynamic frontend.

## 🚀 Projects Included
1. **Cartoonify Image**  
   Upload an image and convert it into a cartoon style.

2. **Customer Sentiment Analysis**  
   Analyze customer sentiment using VADER and RoBERTa models.

3. **News Categorization**  
   Categorize news articles using a pre-trained Naive Bayes model.

4. **OCR Document Parser**  
   Upload and extract text and structured data from documents using OCR.

5. **Pneumonia Detection**  
   Detect pneumonia from chest X-ray images using a CNN model.

---

## 📂 Project Structure
your_project/
├── app/
│ ├── main.py # FastAPI combined backend
│ ├── templates/ # All HTML files
│ ├── parser.py # OCR document parser
│ └── saver.py # JSON saver for OCR
├── static/ # CSS, JS, uploaded files
├── models/ # All ML models
├── uploads/ # Uploaded OCR files
├── data/ # Parsed OCR JSON files
├── requirements.txt
├── Dockerfile (optional)
└── README.md

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd your_project
2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the App
bash
Copy
Edit
uvicorn app.main:app --reload
Access the app at: http://localhost:8000
