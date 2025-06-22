from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import fitz # PyMuPDF for PDF text extraction (better than pdfminer for simplicity)
from docx import Document # For Word text extraction
import re # For text searching
from sklearn.feature_extraction.text import TfidfVectorizer # For classification
from sklearn.naive_bayes import MultinomialNB # A simple classifier
import joblib # To save/load the trained model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Dummy data for demonstration and initial setup ---
# In a real app, this would come from a database
documents_metadata = [
    {"id": 1, "title": "Software Engineering Best Practices", "text": "This document covers software engineering methodologies, agile development, and testing practices."},
    {"id": 2, "title": "Data Analytics in Python", "text": "An introduction to data analysis using Python, pandas, and scikit-learn for machine learning."},
    {"id": 3, "title": "Project Management Principles", "text": "Key concepts in project management, including planning, execution, and risk assessment."},
    {"id": 4, "title": "Machine Learning Algorithms", "text": "Exploring various machine learning algorithms like Naive Bayes, SVM, and decision trees."},
]

# --- Dummy Model Training (for demonstration) ---
# In a real app, you would train this with real data and save it.
# We'll re-train it simply for initial setup
categories = ["Software", "Data Science", "Project Management"]
training_data = [
    ("Software Engineering Best Practices and Agile Methods", "Software"),
    ("Advanced Data Analysis with Python and Pandas", "Data Science"),
    ("Fundamentals of Project Planning and Risk Management", "Project Management"),
    ("Introduction to Machine Learning Models and Algorithms", "Data Science"),
    ("Software Testing and Quality Assurance", "Software"),
]

vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

def train_model():
    global vectorizer, classifier
    texts = [d[0] for d in training_data]
    labels = [d[1] for d in training_data]
    X_train = vectorizer.fit_transform(texts)
    classifier.fit(X_train, labels)
    print("Dummy model trained.")

train_model() # Train the dummy model on startup

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path) # opens a document
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
    return text

def classify_text(text):
    if not text.strip():
        return "Uncategorized" # Handle empty text
    # Use the already fitted vectorizer
    X_new = vectorizer.transform([text])
    prediction = classifier.predict(X_new)[0]
    return prediction

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', documents=documents_metadata, categories=categories)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_text = ""
        if filename.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.endswith('.docx'):
            extracted_text = extract_text_from_docx(filepath)

        # Extract title (simple approach: first line or filename)
        title = extracted_text.split('\n')[0].strip() if extracted_text.strip() else filename

        # Classify the document
        category = classify_text(extracted_text)

        # In a real app, save to Deta Base/DB
        new_id = len(documents_metadata) + 1
        documents_metadata.append({
            "id": new_id,
            "title": title,
            "text": extracted_text,
            "category": category, # Add category here
            "filename": filename
        })

        return redirect(url_for('index'))
    return 'File type not allowed!', 400

@app.route('/document/<int:doc_id>')
def view_document(doc_id):
    doc = next((d for d in documents_metadata if d['id'] == doc_id), None)
    if doc:
        return render_template('document.html', doc=doc)
    return 'Document not found', 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/search', methods=['GET'])
def search_documents():
    query = request.args.get('q', '').lower()
    if not query:
        return redirect(url_for('index'))

    search_results = []
    for doc in documents_metadata:
        if query in doc['title'].lower() or query in doc['text'].lower():
            search_results.append(doc)
    return render_template('index.html', documents=search_results, search_query=query, categories=categories)

@app.route('/category/<category_name>')
def filter_by_category(category_name):
    filtered_docs = [doc for doc in documents_metadata if doc.get('category') == category_name]
    return render_template('index.html', documents=filtered_docs, selected_category=category_name, categories=categories)


if __name__ == '__main__':
    app.run(debug=True)