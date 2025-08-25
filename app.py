from unittest import result
from flask import Flask, render_template, request, jsonify 
import os
from werkzeug.utils import secure_filename

# Import your RAG bot function
from RAG.docBot import run_query

app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), "user_docs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        return f"File {filename} uploaded successfully!", 200

    return "Invalid file type", 400

# ✅ New route for chatbot queries
@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"response": "Please type something."})
    
    # Call your RAG bot
    result = run_query(user_message)

    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True,)
    
