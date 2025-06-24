from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import fitz # PyMuPDF for PDF text extraction
from docx import Document # For Word text extraction
import re # For text searching and highlighting
from sklearn.feature_extraction.text import TfidfVectorizer # CORRECTED: sklearn instead of sklean
from sklearn.naive_bayes import MultinomialNB # A simple classifier
import joblib # To save/load the trained model
import requests # For downloading files from URLs
from urllib.parse import urlparse # To parse URLs and get filenames
import time # For basic performance measurement
import tempfile # For temporary file handling

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' 
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.secret_key = '93f031deb123c4290feca241d5dd6b8a867f7466a1162d34' # المفتاح السري الذي قمت بتحديده

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Dummy data for demonstration and initial setup ---
documents_metadata = [
    {"id": 1, "title": "Software Engineering Best Practices", "text": "This document covers software engineering methodologies, agile development, and testing practices. Software quality is key.", "category": "Software", "filename": "dummy_software.pdf"},
    {"id": 2, "title": "Data Analytics in Python", "text": "An introduction to data analysis using Python, pandas, and scikit-learn for machine learning. Data visualization is crucial.", "category": "Data Science", "filename": "dummy_data_analytics.pdf"},
    {"id": 3, "title": "Project Management Principles", "text": "Key concepts in project management, including planning, execution, and risk assessment. Effective project planning ensures success.", "category": "Project Management", "filename": "dummy_project_mgmt.pdf"},
    {"id": 4, "title": "Machine Learning Algorithms", "text": "Exploring various machine learning algorithms like Naive Bayes, SVM, and decision trees. Algorithms are fundamental.", "category": "Data Science", "filename": "dummy_ml_algorithms.pdf"},
]

# --- Dummy Model Training (for demonstration) ---
categories = ["Software", "Data Science", "Project Management", "Uncategorized"]
training_data = [
    ("Software Engineering Best Practices and Agile Methods", "Software"),
    ("Advanced Data Analysis with Python and Pandas", "Data Science"),
    ("Fundamentals of Project Planning and Risk Management", "Project Management"),
    ("Introduction to Machine Learning Models and Algorithms", "Data Science"),
    ("Software Testing and Quality Assurance", "Software"),
    ("Building Scalable Web Applications", "Software"),
    ("Statistical Modeling and Data Visualization", "Data Science"),
    ("Agile Project Management Methodologies", "Project Management"),
    ("Data Structures and Algorithms for Efficient Programming", "Software"),
    ("Big Data Technologies and Cloud Computing", "Data Science"),
    ("Cloud Infrastructure and Deployment Strategies", "Software"),
    ("Artificial Intelligence and Neural Networks", "Data Science"),
]

# These are initialized globally
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

def train_model():
    """Trains a dummy classification model based on predefined data."""
    # Declare global variables at the very beginning if they might be reassigned
    global vectorizer, classifier 

    texts = [d[0] for d in training_data]
    labels = [d[1] for d in training_data]

    if not texts or not labels:
        print("Warning: Training data is empty or insufficient. Classification may not work.")
        return

    try:
        vectorizer.fit(texts)
        X_train = vectorizer.transform(texts)
        classifier.fit(X_train, labels)
        print("Dummy model trained successfully.")
    except Exception as e:
        print(f"Error training dummy model: {e}")
        # Re-initialize global variables in case of error
        vectorizer = TfidfVectorizer()
        classifier = MultinomialNB()

train_model()

# --- Helper Functions for File Operations and Text Processing ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_drive_file_id(drive_link):
    """Extracts Google Drive file ID from a URL."""
    # This is a very basic extraction and might need more robust handling
    match = re.search(r'(?:id=)([a-zA-Z0-9_-]+)', drive_link)
    if match:
        return match.group(1)
    match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', drive_link)
    if match:
        return match.group(1)
    return None

def get_filename_from_url(url):
    """Extracts a filename from a URL, handling query parameters and ensuring an extension."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)

    if not filename:
        if parsed_url.netloc:
            filename = parsed_url.netloc.replace('.', '_') + '.pdf'
        else:
            filename = "downloaded_file.pdf"

    if '.' not in filename:
        filename += '.pdf'

    return filename

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        document.close()
    except Exception as e:
        print(f"Error extracting PDF text from {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX text from {docx_path}: {e}")
    return text

def classify_text(text):
    """Classifies text into predefined categories using the trained model."""
    if not text.strip():
        return "Uncategorized"

    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_ or \
       not hasattr(classifier, 'classes_') or len(classifier.classes_) == 0:
        print("Warning: Model not trained or components missing. Returning 'Uncategorized'.")
        return "Uncategorized"

    try:
        X_new = vectorizer.transform([text])
        prediction = classifier.predict(X_new)[0]
        return prediction
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Uncategorized"

def highlight_text(text, query):
    """Highlights all occurrences of the query in the text."""
    if not query or not text:
        return text
    return re.sub(r'({})'.format(re.escape(query)), r'<mark>\1</mark>', text, flags=re.IGNORECASE)

# --- Routes ---
@app.route('/')
def index():
    """Renders the main page with document list, search, and category filters."""
    return render_template('index.html', documents=documents_metadata, categories=categories,
                           search_query=request.args.get('q', ''),
                           selected_category=request.args.get('category_name', ''))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Processes either an uploaded file or a file from a given URL."""
    file = request.files.get('file')
    drive_link = request.form.get('drive_link')

    filename = ""
    extracted_text = ""
    filepath = None
    temp_file = None # Initialize temp_file to None

    if file and file.filename and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            flash(f'File "{filename}" uploaded successfully. Processing...', 'info')
        except Exception as e:
            flash(f'Error saving uploaded file: {e}', 'error')
            return redirect(url_for('index'))

        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(filepath)

    elif drive_link and "drive.google.com" in drive_link:
        file_id = extract_drive_file_id(drive_link)
        if not file_id:
            flash("Invalid Google Drive link provided. Could not extract file ID.", 'error')
            return redirect(url_for('index'))

        gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        if '.pdf' in drive_link.lower():
            filename = f"drive_{file_id}.pdf"
        elif '.docx' in drive_link.lower():
            filename = f"drive_{file_id}.docx"
        else:
            flash("Unsupported file type from Google Drive link. Only PDF and DOCX are supported.", 'error')
            return redirect(url_for('index'))

        try:
            response = requests.get(gdrive_url, stream=True, timeout=60)
            response.raise_for_status()

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file.close()
            filepath = temp_file.name

            flash(f'Google Drive file "{filename}" downloaded. Processing...', 'info')

            if filename.lower().endswith('.pdf'):
                extracted_text = extract_text_from_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                extracted_text = extract_text_from_docx(filepath)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file from URL: {drive_link} - {e}")
            flash(f"Error downloading file: {e}. Make sure the link is direct and accessible (publicly shared).", 'error')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"Error processing Google Drive file {filename}: {e}")
            flash(f"Error processing Google Drive file: {e}. Check file content or format.", 'error')
            return redirect(url_for('index'))
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                    print(f"Removed temporary file: {temp_file.name}")
                except Exception as e:
                    print(f"Error removing temporary file {temp_file.name}: {e}")

    else:
        flash("Please upload a file or provide a valid Google Drive link.", 'error')
        return redirect(url_for('index'))

    title = ""
    if extracted_text.strip():
        lines = extracted_text.split('\n')
        for line in lines:
            if line.strip():
                title = line.strip()
                break
    if not title:
        title = filename

    category = classify_text(extracted_text)

    new_id = len(documents_metadata) + 1
    documents_metadata.append({
        "id": new_id,
        "title": title,
        "text": extracted_text,
        "category": category,
        "filename": filename
    })
    flash('Document processed and added successfully!', 'success')

    # Cleanup the locally uploaded file after processing if it's not a temp Google Drive file
    if filepath and os.path.exists(filepath) and not (temp_file and filepath == temp_file.name):
        try:
            os.remove(filepath)
            print(f"Removed uploaded file: {filepath}")
        except Exception as e:
            print(f"Error removing uploaded file {filepath}: {e}")

    return redirect(url_for('index')) # Single return statement for success/final redirect

@app.route('/document/<int:doc_id>')
def view_document(doc_id):
    """Renders the document details page for a specific document."""
    doc = next((d for d in documents_metadata if d['id'] == doc_id), None)
    if doc:
        search_query = request.args.get('highlight_q', '').lower()
        display_text = doc['text']
        if search_query:
            display_text = highlight_text(doc['text'], search_query)
        return render_template('document.html', doc=doc, display_text=display_text)
    flash('Document not found.', 'error')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Allows downloading of original files (only if they exist locally, which is temporary)."""
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    else:
        flash(f"File '{filename}' not found for direct download. It might have been processed and removed.", 'error')
        return redirect(url_for('index'))

@app.route('/search')
def search_documents():
    """Searches documents based on a query and displays search results."""
    start_time = time.time()
    query = request.args.get('q', '').lower()
    
    search_results = []
    if query:
        for doc in documents_metadata:
            if query in doc['title'].lower() or query in doc['text'].lower():
                search_results.append(doc)
    else:
        search_results = documents_metadata

    end_time = time.time()
    search_time_ms = round((end_time - start_time) * 1000, 2)

    flash(f"Search completed in {search_time_ms} ms. Found {len(search_results)} documents.", 'info')
    return render_template('index.html', documents=search_results, search_query=query, categories=categories)

@app.route('/category/<category_name>')
def filter_by_category(category_name):
    """Filters documents by category and displays results."""
    start_time = time.time()
    if category_name == 'All':
        filtered_docs = documents_metadata
        flash("Displaying all documents.", 'info')
    else:
        filtered_docs = [doc for doc in documents_metadata if doc.get('category') == category_name]
        flash(f"Filtered by category '{category_name}'. Found {len(filtered_docs)} documents.", 'info')
    
    end_time = time.time()
    filter_time_ms = round((end_time - start_time) * 1000, 2)

    flash(f"Filter completed in {filter_time_ms} ms.", 'info')
    return render_template('index.html', documents=filtered_docs, selected_category=category_name, categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
