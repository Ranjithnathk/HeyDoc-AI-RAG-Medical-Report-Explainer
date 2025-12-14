# HeyDoc-AI-RAG-Medical-Report-Explainer

HeyDocAI is a Generative AI application designed to help patients and non-experts understand complex radiology reports using **Retrieval-Augmented Generation (RAG)** and **Prompt Engineering**.  
It provides plain-English explanations, structured extractions, and evidence-cited answers sourced from trusted radiology references.

> **NOTE:** This tool is for **educational purposes only** and does **not** provide medical diagnosis or treatment advice.

---

## Key Features

### Explain Radiology Reports
- Converts technical radiology language into:
  - **Simple**
  - **Normal**
  - **Clinician-level** explanations
- Includes **evidence-based citations** for transparency

### Structured Information Extraction
- Extracts report content into JSON:
  - Findings
  - Impression
  - Key terms
- Displays extracted data in readable tables

### Evidence-Backed Q&A (RAG)
- Ask natural-language questions about the report
- Answers are generated **only when sufficient evidence is retrieved**
- Each answer includes **source citations (PDF and its page number)**

### Prompt Engineering & Guardrails
- Input validation (empty, too short, invalid questions)
- Low-confidence retrieval handling (“I don’t know” responses)
- Mandatory disclaimer enforcement
- Context management for multi-turn Q&A

### Evaluation & Metrics
- Latency measurement
- Citation coverage
- Evidence availability rate
- Results saved as structured JSON

---

## Core Generative AI Techniques Used

| Technique | Description |
|---------|-------------|
| Prompt Engineering | Multi-template prompts for explanation, extraction, and Q&A |
| Retrieval-Augmented Generation (RAG) | External medical knowledge retrieved from vector DB |
| Context Management | Injects retrieved evidence + chat history into prompts |
| Guardrails | Prevents hallucination and unsupported answers |

---

## System Architecture

![System_Architecture](https://github.com/Ranjithnathk/HeyDoc-AI-RAG-Medical-Report-Explainer/tree/main/docs/architecture.png)

---

## Knowledge Base

- Curated **radiology teaching PDFs**
- Topics include:
  - Chest X-ray interpretation
  - Thoracic imaging glossary
  - Lung opacities, effusion, pneumothorax
- Documents are chunked and embedded using OpenAI embeddings
- Stored and retrieved using **Pinecone Vector Database**

---

## Tech Stack

| Layer | Technology |
|-----|------------|
| UI | Streamlit |
| LLM | OpenAI GPT models |
| Embeddings | OpenAI Embeddings |
| Vector DB | Pinecone |
| Language | Python |
| Evaluation | Custom metrics scripts |

---

## Project Structure

```bash
├── app/
│ ├── app.py # Streamlit application
│ ├── prompts.py 
│ ├── generate.py 
│ ├── guards.py 
│ └── context.py 
│
├── rag/
│ ├── build_pinecone_index.py
│ ├── chunking.py
│ ├── citations.py
│ ├── embeddings.py
│ ├── loaders.py
│ ├── pinecone_smoke_test.py
│ ├── ranking.py
│ ├── pretriever.py
│ └── pinecone_upsert.py
│
├── data/
│ └── knowledge_base/ # Radiology PDFs
│
├── eval/
│ ├── eval_set.json # Standard evaluation questions
│ ├── run_eval.py # Metrics computation
│ ├── results.json # Evaluation output
│ └── examples/ # Logs & evidence
│
├── docs/
│ ├── architecture.png # System diagram
│ ├── screens/ # UI screenshots
│ └── logo.png # App logo
│
├── tests/
│ └── test_*.py # Component tests
│
├── requirements.txt
├── README.md
└── .env
```

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd HeyDoc-AI-RAG-Medical-Report-Explainer
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a .env file:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=heydocai-medkb
```

### 5. Run the Application
```bash
streamlit run app/app.py
```

---

## Evaluation

Run automated evaluation:
```bash
python -m eval.run_eval
```

**Metrics computed:**
- Average latency
- Citation coverage
- Evidence availability
- Citation coverage given evidence

**Results saved in:**
```bash
eval/results.json
```

---

## Screenshots

**Screenshots demonstrating:**
- Explain tab
- Extract tab
- Evidence Q&A
- Guardrails
- Retrieval settings

---

## Location:

```bash
docs/screens/
```
**Note:** Some outputs span multiple screens and are split using suffixes (like 03a/03b, 05a/05b).

---

## Ethical Considerations
- No personal health data stored
- No medical diagnosis provided
- Bias minimized by grounding answers in trusted medical sources
- Clear disclaimers enforced at all times

---

## Future Improvements
- Multimodal input (X-ray images + text)
- More document domains (MRI, CT, ultrasound)
- Improved citation ranking
- Deployment to cloud hosting

---

## Author

Ranjithnath Karunanidhi
Graduate Student - Information Systems
Northeastern University, Boston

---