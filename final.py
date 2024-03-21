import tkinter as tk
from tkinter import filedialog, messagebox
import textract
from transformers import GPT2TokenizerFast
from PyPDF2 import PdfWriter, PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import warnings
import os
import Api_key
import random

warnings.simplefilter("ignore")
os.environ["OPENAI_API_KEY"] = Api_key.API_KEY

# Initialize global variables
db = None
conversation_history = []


# Function to update the database
def update_database():
    global db
    file_path = 'Data_update/merged_file.pdf'
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    chunks = pages
    doc = textract.process(file_path)

    with open('../../attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
        f.write(doc.decode('utf-8'))

    with open('../../attention_is_all_you_need.txt', 'r') as f:
        text = f.read()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens,
    )

    chunks = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)


# Function to handle AI call
def call(query):
    chat_history = []
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    docs = db.similarity_search(query)
    chain.run(input_documents=docs, question=query)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

    if "ticket" in query.lower():
        ticket_response = handle_ticket_query(query)
        return ticket_response
    else:
        result = qa({"question": query, "chat_history": chat_history})
        return result["answer"]


# Function to handle ticket query
def handle_ticket_query(query):
    ticket_number = create_ticket(query)
    return f"Ticket created successfully. Your ticket number is {ticket_number}"


# Function to create a random ticket number
def create_ticket(query):
    return random.randint(1000, 9999)


# Function to submit user query
def submit_query():
    user_query = entry.get()
    answer = call(user_query)
    conversation_history.append(f"User: {user_query}\nAI: {answer}\n")
    display_conversation()
    entry.delete(0, tk.END)


# Function to display conversation history
def display_conversation():
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    for line in conversation_history:
        text_widget.insert(tk.END, line)
    text_widget.config(state=tk.DISABLED)


# Function to open a new window for PDF upload
def open_new_window():
    new_window = tk.Toplevel(root)
    new_window.title("PDF Upload")
    new_window.geometry("400x200")
    new_window.resizable(False, False)

    def upload_pdf():
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            save_path = os.path.join("Data", os.path.basename(file_path))
            try:
                import shutil
                shutil.copy(file_path, save_path)
                folder_path = 'Data'
                output_folder = 'Data_update'
                output_filename = 'merged_file.pdf'
                writer = PdfWriter()
                for filename in os.listdir(folder_path):
                    if filename.endswith('.pdf'):
                        filepath = os.path.join(folder_path, filename)
                        reader = PdfReader(filepath)
                        for page in reader.pages:
                            writer.add_page(page)

                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
                update_database()
                messagebox.showinfo("Success", f"PDF file saved to: {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Error saving PDF file: {e}")

    upload_button = tk.Button(new_window, text="Upload PDF", command=upload_pdf)
    upload_button.pack(pady=20)


# Function to perform additional task (PDF file manager)
def perform_additional_task():
    def load_files():
        folder_path = "Data"
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", "Invalid folder path")
            return

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        file_listbox.delete(0, tk.END)
        for pdf_file in pdf_files:
            file_listbox.insert(tk.END, pdf_file)

    def delete_file():
        selected_index = file_listbox.curselection()
        if not selected_index:
            messagebox.showerror("Error", "No file selected")
            return

        selected_file = file_listbox.get(selected_index)
        folder_path = "Data"
        file_path = os.path.join(folder_path, selected_file)
        try:
            os.remove(file_path)

            folder_path = 'Data'
            output_folder = 'Data_update'
            output_filename = 'merged_file.pdf'
            writer = PdfWriter()
            for filename in os.listdir(folder_path):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(folder_path, filename)
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        writer.add_page(page)

            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

            update_database()
            messagebox.showinfo("Success", f"File '{selected_file}' deleted successfully")

            load_files()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete file: {str(e)}")

    root = tk.Tk()
    root.title("PDF File Manager")
    root.geometry("380x300")
    root.resizable(False, False)

    load_button = tk.Button(root, text="Load Files", command=load_files)
    load_button.grid(row=0, column=0, padx=5, pady=5)

    file_listbox = tk.Listbox(root, width=60, height=10)
    file_listbox.grid(row=1, column=0, padx=5, pady=5)

    delete_button = tk.Button(root, text="Delete Selected", command=delete_file)
    delete_button.grid(row=2, column=0, padx=5, pady=5)




# Function to handle additional task button click
def handle_additional_task():
    perform_additional_task()


# Tkinter GUI setup
root = tk.Tk()
root.title("Chat with AI")
root.geometry("700x500")
root.resizable(False, False)

entry = tk.Entry(root, width=50)
entry.pack(pady=(10, 0))  # Add top padding

# Submit button with padding
submit_button = tk.Button(root, text="Submit", command=submit_query)
submit_button.pack(pady=5)  # Add padding between button and entry widget

# Open new window button
new_window_button = tk.Button(root, text="Open PDF Upload", command=open_new_window)
new_window_button.pack(pady=5)

# Additional task button
additional_task_button = tk.Button(root, text="Delete PDF", command=handle_additional_task)
additional_task_button.pack(pady=5)  # Add padding between buttons

# Conversation display with flexible width
text_widget = tk.Text(root, height=20, padx=10, pady=10)  # Remove width
text_widget.pack(expand=True, fill="both")  # Allow widget to expand in both directions

root.bind("<Return>", lambda event=None: submit_query())

conversation_history = []

# Update database initially
update_database()

root.mainloop()
