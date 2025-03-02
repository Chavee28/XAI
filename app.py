from flask import Flask, render_template, request, jsonify
import os
import json
from resumeparser import extract_text_from_pdf, parse_resume, prepare_text_for_matching, match_resume
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
PARSED_FOLDER = 'parsed_results'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, PARSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PARSED_FOLDER'] = PARSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of available parsed resumes
    parsed_files = [f for f in os.listdir(PARSED_FOLDER) if f.endswith('.json')]
    return render_template('index.html', parsed_files=parsed_files)

@app.route('/parse', methods=['GET', 'POST'])
def parse():
    if request.method == 'GET':
        return render_template('parse.html')
    
    if request.method == 'POST':
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # First extract text from PDF
                resume_text = extract_text_from_pdf(filepath)
                if not resume_text:
                    return jsonify({'error': 'Could not extract text from PDF'}), 500
                
                # Then parse the extracted text
                parsed_data = parse_resume(resume_text)
                if not parsed_data:
                    return jsonify({'error': 'Could not parse resume text'}), 500
                
                # Convert string response to JSON if needed
                try:
                    if isinstance(parsed_data, str):
                        parsed_data = json.loads(parsed_data)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON format in parsed data'}), 500
                
                # Save parsed data to JSON
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_filepath = os.path.join(app.config['PARSED_FOLDER'], json_filename)
                
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, indent=2)
                
                # Debug prints
                print("Extracted Text:", resume_text[:500])  # First 500 chars
                print("Parsed Data:", parsed_data)
                
                return jsonify({
                    'result': json.dumps(parsed_data, indent=2),
                    'filename': json_filename
                })
                
            except Exception as e:
                print(f"Error in parse route: {e}")  # Debug print
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400

@app.route('/get_resumes', methods=['GET'])
def get_resumes():
    try:
        # Get list of parsed resume files
        resumes = [f for f in os.listdir(PARSED_FOLDER) if f.endswith('.json')]
        return jsonify({'resumes': resumes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['GET', 'POST'])
def match():
    if request.method == 'GET':
        return render_template('match.html')
        
    try:
        data = request.json
        resume_file = data.get('resumeFile')
        job_description = data.get('jobDescription')
        
        print("Resume File:", resume_file)  # Debug print
        print("Job Description:", job_description)  # Debug print
        
        if not resume_file or not job_description:
            return jsonify({'error': 'Missing resume file or job description'}), 400
            
        # Read parsed resume
        parsed_filepath = os.path.join(PARSED_FOLDER, resume_file)
        print("Looking for file at:", parsed_filepath)  # Debug print
        
        if not os.path.exists(parsed_filepath):
            available_files = os.listdir(PARSED_FOLDER)
            print("Available files:", available_files)
            return jsonify({'error': f'Resume not found: {resume_file}. Available files: {available_files}'}), 404
            
        try:
            with open(parsed_filepath, 'r', encoding='utf-8') as f:
                parsed_resume = f.read()
                print("Successfully read resume file")  # Debug print
                parsed_data = json.loads(parsed_resume)
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            return jsonify({'error': 'Invalid JSON in resume file'}), 400
        except Exception as e:
            print(f"File reading error: {e}")
            return jsonify({'error': 'Error reading resume file'}), 500
        
        # Prepare resume text for matching
        resume_text = prepare_text_for_matching(parsed_data)
        print("Prepared Resume Text:", resume_text)  # Debug print
        
        if not resume_text:
            return jsonify({'error': 'Failed to prepare resume text'}), 500
        
        # Match resume against job description
        match_result = match_resume(resume_text, job_description)
        print("Match Result:", match_result)  # Debug print
        
        return jsonify(match_result)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 