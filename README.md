# Simple PDF Chatbot

This is a simple Python project I made that lets you ask questions about your PDF files. 

It reads a PDF, breaks the text down into smaller chunks, and uses LLM to answer your questions. The best part is that it ONLY uses the information inside the PDF to answer. If the answer isn't in the PDF, it will tell you it doesn't know!

## How it works
1. **Reads the PDF:** Uses `pymupdf` to grab all the text from the file.
2. **Splits the text:** Chops the text into smaller blocks so the LLM can process it easily.
3. **Creates Embeddings:** Converts the text chunks into numbers using `sentence-transformers`.
4. **Searches:** Uses `faiss` to find the most relevant chunks of text based on your question.
5. **Answers:** Sends the relevant text to a large language model (Llama 3 via the Groq API) to generate the final answer.

## What you need to do before running

### 1. Add your PDF file
Make sure you have a PDF file you want to ask questions about. **Rename your PDF file to `context.pdf`** and place it in the exact same folder as the `app.py` script. 

### 2. Set up your Groq API Key
This project uses Groq to run the AI model. You will need a API key from them.
- Rename the `.env.example` file to `.env`.
- Open the `.env` file and replace `YOUR_GROQ_API_KEY` with your actual Groq API key. It should look like this:

```sh
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### 3. Install the requirements
Open your terminal or command prompt, make sure you are in the project folder, and run this command to download all the necessary tools:
```bash
pip install -r requirements.txt
```

### 4. Run the program
Once everything is set up, just run the Python script:

```sh
python main.py
```

It will load the PDF, process it, and then ask you to enter a question. When you are done chatting, just type exit or quit to end the program.