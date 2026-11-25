# 📚 Statistical Techniques AI Tutor (RAG Application)

An AI-powered Statistical Techniques Tutor that uses Retrieval-Augmented Generation (RAG) to answer statistics-related questions from PDF documents.

The application retrieves relevant information from uploaded PDF documents and generates context-based answers using NVIDIA's Nemotron language model.

## 🚀 Features

- 📄 PDF Document Loading
- ✂️ Automatic Text Splitting and Chunking
- 🔍 Semantic Search using Vector Embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- ⚡ NVIDIA Nemotron LLM Integration
- 📊 Statistical Question Answering
- 💬 Interactive Streamlit User Interface
- 📌 Retrieved Context for Better Answer Transparency

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| Streamlit | Web Application Interface |
| LangChain | RAG Workflow |
| NVIDIA API | AI Text Generation |
| FastEmbed | Text Embeddings |
| Qdrant | Vector Database |
| PyPDF | PDF Document Loading |

## 📂 Project Structure

```text
My_First_App/
│
├── document_loader/
│   └── STATS.pdf
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hg968062-cmd/My_First_App.git
```

### 2. Navigate to Project Folder

```bash
cd My_First_App
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory.

Add your NVIDIA API key:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Never upload your `.env` file or API key to GitHub.

### 6. Run the Application

```bash
python -m streamlit run main.py
```

The application will open in your browser.

## 🔄 How It Works

1. **Document Loading:** Loads statistical information from PDF documents.
2. **Text Splitting:** Divides large documents into smaller text chunks.
3. **Embedding Generation:** Converts text chunks into vector embeddings.
4. **Vector Storage:** Stores embeddings in an in-memory Qdrant vector store.
5. **Retrieval:** Finds the most relevant document chunks for a question.
6. **AI Response:** NVIDIA Nemotron generates an answer using the retrieved context.

## 🎯 Project Objective

The main objective of this project is to make learning statistical techniques easier through Artificial Intelligence.

Students can ask questions and receive relevant answers based on the provided study material.

## 🔮 Future Enhancements

- Support for multiple PDF documents.
- Chat history and conversation memory.
- Improved document search.
- Additional educational subjects.
- Better response evaluation.

## 👨‍💻 Author

**Himanshu Gupta**

GitHub: https://github.com/hg968062-cmd

## 📜 License

This project is developed for educational and learning purposes.
