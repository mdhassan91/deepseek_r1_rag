# src/evaluate.py
import csv
import json
import time
from pathlib import Path
from typing import List, Dict
import tempfile
import base64
import asyncio
from dataclasses import dataclass  # Add this import
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Paths
TEST_CSV_PATH = Path("questions_data.csv")
RESULTS_PATH = Path("../evaluation/results.json")

# Define the State class directly in evaluate.py
@dataclass
class QA:
    """A question and answer pair."""
    question: str
    answer: str

class State:
    """The app state."""
    chats: List[List[QA]] = [[]]
    base64_pdf: str = ""
    uploading: bool = False
    current_chat: int = 0
    processing: bool = False
    db_path: str = tempfile.mkdtemp()
    pdf_filename: str = ""
    knowledge_base_files: List[str] = []
    upload_status: str = ""

    _query_engine = None
    _temp_dir = None

    def setup_llamaindex(self):
        """Setup LlamaIndex with models and prompt template."""
        if self._query_engine is None and self._temp_dir:
            # Setup LLM
            llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0)
            
            # Setup embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
                trust_remote_code=True
            )
            
            # Configure settings
            Settings.embed_model = embed_model
            Settings.llm = llm

            # Load documents
            loader = SimpleDirectoryReader(
                input_dir=self._temp_dir,
                required_exts=[".pdf"],
                recursive=True
            )
            docs = loader.load_data()

            # Create index and query engine
            index = VectorStoreIndex.from_documents(docs, show_progress=True)
            
            # Setup streaming query engine with custom prompt
            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
            
            self._query_engine = index.as_query_engine(streaming=True)
            self._query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

    async def process_question(self, question: str):
        """Process a question and return the response."""
        if self.processing or not question or not self._query_engine:
            return ""

        self.processing = True
        self.chats[self.current_chat].append(QA(question=question, answer=""))
        
        # Get streaming response from LlamaIndex
        streaming_response = self._query_engine.query(question)
        answer = ""

        # Process the streaming response
        for chunk in streaming_response.response_gen:
            answer += chunk
            self.chats[self.current_chat][-1].answer = answer

        self.processing = False
        return answer

    async def handle_upload(self, file_path: Path):
        """Handle file upload and processing."""
        self.uploading = True
        
        # Read the file
        with open(file_path, "rb") as f:
            upload_data = f.read()
        
        # Create temporary directory if not exists
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
            
        outfile = Path(self._temp_dir) / file_path.name
        self.pdf_filename = file_path.name

        with outfile.open("wb") as file_object:
            file_object.write(upload_data)

        # Base64 encode the PDF content
        self.base64_pdf = base64.b64encode(upload_data).decode('utf-8')

        # Setup LlamaIndex
        self.setup_llamaindex()
        
        self.knowledge_base_files.append(self.pdf_filename)
        self.upload_status = f"Added {self.pdf_filename} to knowledge base"

        self.uploading = False

def load_test_data():
    """Load test data from CSV."""
    test_data = []
    with open(TEST_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                "file_name": row["File Name"],
                "content": "",  # Add PDF content if available
                "questions": [
                    {
                        "question": row["Questions"],
                        "expected_answer": row["Expected Answer"],
                        "actual_answer": "",  # Will be filled during evaluation
                        "is_correct": False,  # Will be filled during evaluation
                        "latency": 0.0,       # Will be filled during evaluation
                        "context_present": row["Context Present"]  # Add context info
                    }
                ]
            })
    return test_data

async def evaluate(test_data):
    """
    Evaluate the chatbot on the test dataset.
    """
    results = []
    total_questions = 0
    correct_answers = 0
    error_handling_success = 0
    total_latency = 0

    # Initialize chatbot state
    state = State()
    state._temp_dir = Path("../evaluation/test_pdfs")  # Point to test PDFs

    for doc in test_data:
        # Upload the PDF
        pdf_path = state._temp_dir / doc["file_name"]
        if pdf_path.exists():
            await state.handle_upload(pdf_path)
        
        # Process each question
        for q in doc["questions"]:
            total_questions += 1

            # Simulate chatbot response
            start_time = time.time()
            actual_answer = await state.process_question(q["question"])
            latency = time.time() - start_time
            total_latency += latency

            # Check if answer is correct
            is_correct = actual_answer.strip().lower() == q["expected_answer"].strip().lower()
            if is_correct:
                correct_answers += 1

            # Check error handling
            if q["expected_answer"].lower() == "i don't know" and actual_answer.lower() == "i don't know":
                error_handling_success += 1

            # Save results
            q["actual_answer"] = actual_answer
            q["is_correct"] = is_correct
            q["latency"] = latency

        results.append(doc)

    # Calculate metrics
    accuracy = correct_answers / total_questions
    avg_latency = total_latency / total_questions
    error_handling_rate = error_handling_success / total_questions

    return {
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "avg_latency": avg_latency,
            "error_handling_rate": error_handling_rate,
        }
    }

if __name__ == "__main__":
    # Load test data
    test_data = load_test_data()

    # Run evaluation
    import asyncio
    evaluation_results = asyncio.run(evaluate(test_data))

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(evaluation_results, f, indent=2)

  
    # Print metrics
    metrics = evaluation_results["metrics"]
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} seconds")
    print(f"Error Handling Rate: {metrics['error_handling_rate']:.2%}")