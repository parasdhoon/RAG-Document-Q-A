# Conversational RAG with PDF Uploads and Chat History

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF files and interact with their content through a conversational interface. The project leverages LangChain for document retrieval and GPT models for conversational AI. It keeps track of chat history, providing a context-aware Q&A system.

## Project Features

- **PDF Uploads**: Users can upload multiple PDF files.
- **Conversational Interface**: Users can ask questions related to the content of the PDFs.
- **Chat History**: The system stores chat history for context-aware responses.
- **RAG Implementation**: Combines document retrieval and GPT-based question answering for detailed responses.

---

## Setup and Installation

To set up the project, follow the steps below. You can create the environment using **`virtualenv`** or **`conda`**.

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-name>
```
### 2. Set Up the Virtual Environment

To ensure dependencies are isolated, it's recommended to set up a virtual environment.

#### Using `virtualenv` (Recommended)

1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

#### Using `conda`

Alternatively, you can create a conda environment:

1. **Create a conda environment**:

    ```bash
    conda create -n rag_project python=3.10
    ```

2. **Activate the conda environment**:

    ```bash
    conda activate rag_project
    ```

---

### 3. Install Dependencies

Before running the project, you'll need to install the required dependencies.

1. **Install the dependencies** using `pip`:

    If the repository includes a `requirements.txt` file, you can install all dependencies in one go:

    ```bash
    pip install -r requirements.txt
    ```

    If the `requirements.txt` file is not provided, manually install the following dependencies:

    ```bash
    pip install streamlit langchain langchain-core langchain-huggingface langchain-chroma langchain-community langchain-groq langchain-text-splitters dotenv
    ```

---

### 4. Set Up Environment Variables

This project requires API keys for **HuggingFace** and **Groq**.

1. Create a `.env` file in the root directory of the project.
2. Add your API keys for HuggingFace and Groq:

    ```plaintext
    HF_TOKEN=<your-huggingface-api-token>
    GROQ_API_KEY=<your-groq-api-key>
    ```

---

### 5. Running the Project

Once your environment is set up and the dependencies are installed, you're ready to run the project.

1. **Start the Streamlit app**:

    To run the application, execute the following command in the terminal:

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

---

### 6. Using the Application

1. **Upload PDFs**: In the Streamlit interface, use the file uploader to upload one or more PDF files. The content from the uploaded PDFs will be processed for retrieval.

2. **Ask Questions**: Once the PDFs are uploaded, enter your question into the input field. The system will use the uploaded PDFs' content to generate context-aware responses.

3. **View Responses**: The system will display a detailed answer based on the content of the uploaded PDFs, along with the chat history that provides context to the current question.

---

## Project Structure

```bash
.
├── app.py               # Main Streamlit application file
├── .env                 # Environment variables for API keys
├── requirements.txt     # Required Python packages
├── temp.pdf             # Temporary storage for PDF files (generated during runtime)
└── chroma/              # Directory for storing Chroma embeddings
```
## Dependencies

This project requires the following Python packages:

- **Streamlit**: A framework to build interactive web applications.
- **LangChain**: A library to create document-based question-answering systems and pipelines.
- **Chroma**: A vector store used for document retrieval and storage of embeddings.
- **HuggingFace Embeddings**: Embedding models from HuggingFace to convert text into vector representations.
- **Groq API**: Integration with Groq-powered models for efficient processing.
- **Python-dotenv**: A library to load environment variables from a `.env` file.

You can install these dependencies using:

```bash
pip install streamlit langchain langchain-core langchain-huggingface langchain-chroma langchain-community langchain-groq langchain-text-splitters dotenv
```
