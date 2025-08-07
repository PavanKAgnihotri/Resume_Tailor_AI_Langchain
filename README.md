# Resume_Tailor_AI_Langchain

**Resume Tailor AI**
This project provides a pipeline to semantically match your resume content against a job description and then uses the Gemini AI model to rewrite the most relevant parts of your resume to better align with the job requirements.

**Prerequisites**
  Python 3.11 or higher
  Access to the Google Gemini API. You will need a Google API key.

LangChain + LangGraph Architecture
This project leverages LangChain and LangGraph to orchestrate a multi-step, intelligent pipeline that dynamically rewrites a user’s resume based on a provided job description (JD). The final output is a tailored PDF resume optimized for both human readability and ATS systems.

What is LangChain?
LangChain is an open-source framework designed to build applications powered by language models. It provides the building blocks for:
  Prompt templates
  Vector stores
  Embedding models
  Memory
  Chains and agents

In this project, we use LangChain to:
  Embed both resume and job description chunks using Google Gemini
  Store and query these embeddings using FAISS vector stores
  Manage and interact with documents and LLMs in a modular way

What is LangGraph?
LangGraph is a powerful extension of LangChain that allows developers to define multi-step workflows using stateful directed graphs.
It provides:
  Graph-based control flow (like DAGs)
  State management across nodes
  Retries, loops, and branching
  Reusability and modular orchestration

How LangGraph Is Used in This Project
The resume tailoring pipeline is implemented as a LangGraph consisting of the following nodes:
              ┌──────────────┐
              │ resume_parser│ ← extracts resume text & links from PDF
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │   jd_parser  │ ← splits JD into semantic chunks
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │   embedder   │ ← embeds resume & JD chunks using Gemini + LangChain
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │   matcher    │ ← compares JD ↔ resume via cosine similarity
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │   rewriter   │ ← rewrites matching resume bullets using Gemini LLM
              └──────┬───────┘
                 
LangChain Components Used
GoogleGenerativeAIEmbeddings: Embeds resume + JD chunks using Gemini
FAISS: Stores and retrieves vector embeddings
Document:	Wraps each text chunk for LangChain processing
RecursiveTextSplitter: Breaks resume/JD into semantically useful chunks
LLM (Gemini 2.5 Flash):	Rewrites resume bullets to match JD tone

Benefits of Using LangChain + LangGraph
Modularity: Each step (e.g., parsing, embedding, rewriting) is isolated and reusable

Scalability: Easily supports multiple resumes and job descriptions

Extensibility: Add new nodes for cover letter generation, grammar checking, interview Q&A, etc.

Resilience: LangGraph handles retry logic and intermediate state recovery

Transparency: You can inspect state at every node for debugging or analytics

**Project Workflow**
The project follows these main steps:

**Step 1**: Resume Ingestion Pipeline
This part of the code processes your resume PDF.

  1. Extract Text: Reads the entire text content from your resume PDF.
  2. Extract Links: Identifies and extracts any external links (like GitHub or LinkedIn profiles) present in the PDF.
  3. Chunk Text with Metadata: Splits the extracted text into smaller, manageable chunks (documents) and attaches the extracted links as metadata to each chunk. This helps in retaining context and including relevant links during the rewriting phase.
  4. Embed Documents: Converts each text chunk into a numerical vector representation (embedding) using the Gemini embedding-001 model. These embeddings capture the semantic meaning of the text. The embeddings are also normalized.
  5. Build and Save Vectorstore: Creates a FAISS vector store from the embedded resume chunks and their corresponding vectors. This vector store allows for efficient similarity search. The vector store is saved locally for later use.

**Step 2**: Job Description Ingestion Pipeline
This part processes the job description provided as a string.

  1. Chunk Job Description: Splits the job description text into smaller chunks.
  2. Embed Job Description: Converts each job description chunk into a numerical vector representation using the same Gemini embedding-001 model.
  3. Build and Save Vectorstore: Creates and saves a separate FAISS vector store for the job description chunks and their embeddings.

**Step 3**: Semantic Matching Engine
This step compares the resume and job description to find the most relevant parts of your resume for the job.

  1. Load Vectorstores: Loads the previously saved FAISS vector stores for both the resume and the job description.
  2. Compute Similarity Matrix: Calculates the cosine similarity between the embedding vectors of all job description chunks and all resume chunks. This matrix indicates how semantically similar each job description chunk is to each resume chunk.
  3. Rank Resume Matches: For each job description chunk, it finds the top k most similar resume chunks based on the similarity matrix. The relevant links associated with the matched resume chunks are also included.

**Step 4**: Resume Rewriter + Humanizer using Gemini (LLM)
This is the core of the resume tailoring process.

  1. Gemini Setup: Initializes the Gemini model (gemini-1.5-flash in this case) for generating rewritten text, using your provided API key.
  2. Prompt Template: Defines a prompt structure that instructs the Gemini model to rewrite a given resume chunk to better match a specific job description chunk, while maintaining bullet point format and optionally including relevant links.
  3. Rewrite Resume Matches: Iterates through the top matches found in the previous step. For each pair of job description chunk and matched resume chunk, it constructs a prompt using the prompt template and sends it to the Gemini model for rewriting. Includes rate limiting to manage API calls.
  4. Display Results: Prints the original resume chunk, the corresponding job description context, the similarity score, and the rewritten resume chunk generated by the Gemini model.

This process provides a tailored version of your resume content based on the specific job description, aiming to highlight the most relevant experiences and skills.
