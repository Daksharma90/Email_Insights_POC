import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import time
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import traceback

# Constants
GROQ_API_KEY = "-----"  # Your Groq API Key here.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS_PER_CHUNK = 80000  # Reduced chunk size to be safer with token limits
MAX_GROQ_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

CHUNK_PROMPT = """You are a factory operations analyst specializing in identifying issues from production-related email logs. Carefully analyze the following email log content and provide a structured, detailed summary that includes:

1. 📌 **Key Issues Identified** – Clearly list the main problems, errors, or concerns raised in this email.
2. ⚠️ **Anomalies or Irregularities** – Highlight any unusual events, deviations, or unexpected patterns mentioned.
3. 🔁 **Emerging Patterns or Recurring Themes** – Note if this email shows signs of recurring issues seen in other logs (if possible).
4. ✅ **Suggested Actions** – Offer specific, actionable recommendations based on the issues and anomalies found.

Be concise but thorough. Focus on what would matter most to a factory supervisor or operations lead.

EMAIL LOG:
\"\"\"
{chunk}
\"\"\"
"""

FINAL_PROMPT = """You are a senior factory insights analyst. Your task is to synthesize a clear, high-level executive summary based on multiple email-level summaries from factory operations. Use the structure below and ensure the summary is detailed, insightful, and actionable.

1. 📌 **Overall Summary** – A brief 2–3 sentence overview of the situation across all emails.
2. 🔍 **Repeated Key Issues** – List and describe the most commonly occurring problems. Group them by theme (e.g., supply chain, machine failure, personnel issues).
3. 📊 **Patterns Across Emails** – Highlight trends, recurring bottlenecks, repeated complaints, or delays.
4. ⚠️ **Notable Anomalies or Outliers** – Point out any critical or unusual incidents that stood out.
5. ✅ **Recommended Next Steps** – Provide practical recommendations to address the key issues and prevent future anomalies.

Make the language clear and precise. The output should be suitable for senior management to make decisions.

EMAIL-LEVEL SUMMARIES:
\"\"\"
{all_chunks_summary}
\"\"\"
"""

def safe_request_post(url, headers, json_payload, max_retries=MAX_GROQ_RETRIES):
    """
    Robustly makes a POST request with retries and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    sleep_time = int(retry_after)
                else:
                    sleep_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"RateLimit: {e}.  Retry in {sleep_time} seconds.  Attempt {attempt + 1}/{max_retries}")
                time.sleep(sleep_time)
            else:
                logger.error(f"HTTPError: {e}.  Status Code: {response.status_code}.  No retry.")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException: {e}.  Attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None

def call_groq(prompt, temperature=0.3, max_tokens=2048):
    """
    Calls the Groq API to get a response for a given prompt.
    """
    try:
        max_prompt_chars = 4 * (4096 - max_tokens)
        if len(prompt) > max_prompt_chars:
            prompt = prompt[-max_prompt_chars:]

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a brilliant analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        res_json = safe_request_post(GROQ_API_URL, HEADERS, payload)
        if res_json and 'choices' in res_json and len(res_json['choices']) > 0:
            return res_json['choices'][0]['message']['content']
        else:
            logger.error(f"Groq API call failed or returned empty 'choices'.  Response: {res_json}")
            return "[Error: Groq API issue or no response choices]"
    except Exception as e:
        logger.error(f"Exception in call_groq: {e}.  Traceback:\n{traceback.format_exc()}")
        return "[Error: Exception during API call]"

def call_rag(query, context):
    """
    Calls the Groq API for a Retrieval Augmented Generation (RAG) query.
    """
    prompt = f"""You are a helpful assistant...
Context:
\"\"\"
{context}
\"\"\"
Question: {query}
Answer:"""
    max_tokens = 1024
    max_prompt_chars = 4 * (4096 - max_tokens)
    if len(prompt) > max_prompt_chars:
        prompt = prompt[-max_prompt_chars:]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a smart RAG-based assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    res_json = safe_request_post(GROQ_API_URL, HEADERS, payload)
    if res_json and 'choices' in res_json and len(res_json['choices']) > 0:
        return res_json['choices'][0]['message']['content']
    else:
        logger.error(f"RAG call failed. Response: {res_json}")
        return "[Error: RAG failed]"

def smart_chunk_emails(emails, max_chars=MAX_CHARS_PER_CHUNK):
    """
    Intelligently chunks emails, handling large emails and combining smaller ones.
    """
    logger.info(f"smart_chunk_emails: Total emails received for chunking: {len(emails)}")
    chunks = []
    current_chunk = ""
    for idx, email in enumerate(emails):
        email_text = email.strip()
        if not email_text:
            continue
        if len(email_text) > max_chars:
            logger.info(f"Email {idx + 1} is larger than max_chars ({len(email_text)} chars), splitting...")
            for i in range(0, len(email_text), max_chars):
                part = email_text[i:i + max_chars]
                chunks.append(part)
            continue

        if len(current_chunk) + len(email_text) + 5 <= max_chars:
            current_chunk += email_text + "\n---\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = email_text + "\n---\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.info(f"smart_chunk_emails: Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i + 1} size: {len(chunk)} characters")
    return chunks

class EmailInsightsApp:
    """
    Main application class for the Email Insights & Smart Q&A tool.
    """
    def __init__(self, root):
        """
        Initializes the main application window and its components.
        """
        self.root = root
        self.root.title("📧 Email Insights & Smart Q&A")
        self.root.geometry("900x720")
        self.root.configure(bg="#1e1e2f")

        self.file_path = None
        self.emails = None
        self.faiss_index = None
        self.embed_model = None

        self.batch_summaries = []
        self.current_batch_index = None
        self.summary_buttons = []  # To store the batch summary buttons
        self.executive_summary_button = None # To store executive summary button

        header = tk.Label(root, text="📧 Email Insights & Smart Q&A", font=("Segoe UI", 22, "bold"), fg="#00ffcc", bg="#1e1e2f")
        header.pack(pady=(12, 10))

        top_frame = tk.Frame(root, bg="#2a2a40")
        top_frame.pack(fill="x", padx=15, pady=5)

        upload_btn = tk.Button(top_frame, text="📂 Upload emails_cleaned.txt", font=("Segoe UI", 12, "bold"),
                                 bg="#00b894", fg="white", activebackground="#019875", command=self.upload_file)
        upload_btn.pack(side="left", padx=10, pady=10)

        self.summary_status = tk.Label(top_frame, text="No file loaded", font=("Segoe UI", 11), fg="#cccccc", bg="#2a2a40")
        self.summary_status.pack(side="left", padx=20)

        self.button_frame = tk.Frame(root, bg="#2a2a40")  # Separate frame for buttons
        self.button_frame.pack(fill="x", padx=15, pady=5)

        self.progress_summary = tk.DoubleVar()
        self.progress_embedding = tk.DoubleVar()
        self.summary_pb = ttk.Progressbar(root, variable=self.progress_summary, maximum=100)
        self.embedding_pb = ttk.Progressbar(root, variable=self.progress_embedding, maximum=100)
        self.summary_pb.pack(fill="x", padx=15, pady=5)
        self.embedding_pb.pack(fill="x", padx=15, pady=(0, 10))

        summary_label = tk.Label(root, text="📝 Summary Output", font=("Segoe UI", 14, "bold"),
                                  fg="#00d8ff", bg="#1e1e2f")
        summary_label.pack(anchor="w", padx=20)
        self.summary_output = scrolledtext.ScrolledText(root, height=12, font=("Consolas", 11),
                                                 bg="#121223", fg="#b3f3ff", insertbackground="white")
        self.summary_output.pack(fill="both", expand=False, padx=20, pady=(0, 15))

        query_frame = tk.Frame(root, bg="#2a2a40")
        query_frame.pack(fill="x", padx=15, pady=5)

        query_label = tk.Label(query_frame, text="🔎 Ask a question about the emails:", font=("Segoe UI", 13),
                                 fg="#eeeeee", bg="#2a2a40")
        query_label.pack(anchor="w", pady=(0, 5))

        self.query_entry = tk.Entry(query_frame, font=("Segoe UI", 12), width=70)
        self.query_entry.pack(side="left", padx=(0, 10), pady=5)

        ask_btn = tk.Button(query_frame, text="💬 Get Answer", font=("Segoe UI", 12, "bold"),
                                 bg="#0984e3", fg="white", activebackground="#0652dd", command=self.ask_question)
        ask_btn.pack(side="left")

        self.answer_output = scrolledtext.ScrolledText(root, height=10, font=("Consolas", 11),
                                                 bg="#0b1220", fg="#d8f2ff", insertbackground="white")
        self.answer_output.pack(fill="both", expand=True, padx=20, pady=(10, 20))

    def upload_file(self):
        """
        Opens a file dialog to select an email text file, loads the emails,
        and starts the processing threads.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.emails = content.split('---\n')
            logger.info(f"upload_file: Total emails loaded: {len(self.emails)}")
            self.summary_status.config(text=f"{len(self.emails)} emails loaded")

            self.chunks = smart_chunk_emails(self.emails)
            self.batch_summaries = [None] * len(self.chunks)
            logger.info(f"upload_file: Total chunks after smart_chunk_emails: {len(self.chunks)}")
            self.create_summary_buttons()  # Create buttons here

            threading.Thread(target=self.process_batches_thread, daemon=True).start()
            threading.Thread(target=self.process_embedding, daemon=True).start()
        except Exception as e:
            logger.error(f"Error reading file: {e}.  Traceback:\n{traceback.format_exc()}")
            self.summary_status.config(text=f"Error: {e}")

    def create_summary_buttons(self):
        """
        Creates the batch summary buttons and the executive summary button.
        """
        # Clear any existing buttons
        for button in self.summary_buttons:
            button.destroy()
        self.summary_buttons = []
        if self.executive_summary_button:
            self.executive_summary_button.destroy()

        # Create buttons for each batch
        for i in range(len(self.chunks)):
            btn = tk.Button(self.button_frame, text=f"Batch {i + 1}", font=("Segoe UI", 10, "bold"),
                            bg="#0984e3", fg="white", width=8,
                            command=lambda idx=i: self.display_batch_summary(idx))
            btn.pack(side="left", padx=4, pady=4)
            self.summary_buttons.append(btn)  # Store the button

        # Create button for executive summary
        self.executive_summary_button = tk.Button(self.button_frame, text="Exec. Summary", font=("Segoe UI", 10, "bold"),
                            bg="#4CAF50", fg="white", width=12,
                            command=self.display_executive_summary)
        self.executive_summary_button.pack(side="left", padx=15, pady=4)

    def display_batch_summary(self, batch_idx):
        """
        Displays the summary for a selected batch in the text widget.
        """
        self.current_batch_index = batch_idx
        summary = self.batch_summaries[batch_idx]
        self.summary_output.delete(1.0, tk.END)
        if summary is None:
            self.summary_output.insert(tk.END, "Processing the summary of this batch, please wait...")
        else:
            self.summary_output.insert(tk.END, summary)

    def display_executive_summary(self):
        """
        Displays the final executive summary.
        """
        if hasattr(self, 'final_summary'): # Check if the final summary has been generated
            self.summary_output.delete(1.0, tk.END)
            self.summary_output.insert(tk.END, "=== Executive Summary ===\n\n" + self.final_summary)
        else:
            self.summary_output.delete(1.0, tk.END)
            self.summary_output.insert(tk.END, "Processing final summary, please wait...")

    def process_batches_thread(self):
        """
        Processes each chunk of emails to generate summaries using the Groq API.
        This function runs in a separate thread.
        """
        for i, chunk in enumerate(self.chunks):
            if self.batch_summaries[i] is None:
                prompt = CHUNK_PROMPT.format(chunk=chunk)
                logger.info(f"Processing batch {i + 1}/{len(self.chunks)} ...")
                summary = call_groq(prompt)
                self.batch_summaries[i] = summary
                logger.info(f"Batch {i + 1} summary obtained.")
                self.progress_summary.set(((i + 1) / len(self.chunks)) * 100)
                if self.current_batch_index == i:
                    self.display_batch_summary(i)
                time.sleep(60)  # Wait 60 seconds to avoid rate limits

        # After all batches, create final executive summary
        logger.info("Generating final executive summary from all batch summaries...")
        all_summaries_text = "\n\n".join([s for s in self.batch_summaries if s])
        final_prompt = FINAL_PROMPT.format(all_chunks_summary=all_summaries_text)
        final_summary = call_groq(final_prompt)
        self.final_summary = final_summary
        logger.info("Final executive summary generated.")
        self.progress_summary.set(100)
        self.display_executive_summary() # Display it automatically

    def process_embedding(self):
        """
        Computes embeddings for the email chunks using SentenceTransformer
        and builds a FAISS index for similarity search.  Runs in a thread.
        """
        try:
            # Create sentence transformer model for embeddings
            logger.info("Loading embedding model...")
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("Computing embeddings for chunks...")
            chunk_embeddings = self.embed_model.encode(self.chunks, show_progress_bar=True)
            dimension = chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(chunk_embeddings).astype('float32'))
            logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors.")
            self.progress_embedding.set(100)
        except Exception as e:
            logger.error(f"Error creating embeddings or FAISS index: {e}. Traceback:\n{traceback.format_exc()}")
            self.summary_status.config(text=f"Error: Embedding/Indexing failed.  Check console.")
            self.faiss_index = None

    def ask_question(self):
        """
        Handles user questions about the emails by performing a similarity search
        on the embeddings and using the Groq API to generate an answer.
        """
        query = self.query_entry.get().strip()
        if not query:
            return
        if self.faiss_index is None or self.embed_model is None:
            self.answer_output.delete(1.0, tk.END)
            self.answer_output.insert(tk.END, "Index not ready. Please wait until embedding processing is done.")
            return

        try:
            query_embedding = self.embed_model.encode([query]).astype('float32')
            D, I = self.faiss_index.search(query_embedding, k=3)
            relevant_chunks = [self.chunks[i] for i in I[0]]

            context = "\n\n".join(relevant_chunks)
            answer = call_rag(query, context)
            self.answer_output.delete(1.0, tk.END)
            self.answer_output.insert(tk.END, answer)
        except Exception as e:
            logger.error(f"Error answering question: {e}. Traceback:\n{traceback.format_exc()}")
            self.answer_output.delete(1.0, tk.END)
            self.answer_output.insert(tk.END, f"Error: {e}.  Check the console for details.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailInsightsApp(root)
    root.mainloop()
