# FOREO AI Assistant - RAG Chatbot with Reasoning Loop

An intelligent customer support chatbot for FOREO products that combines Retrieval-Augmented Generation (RAG) with a reasoning loop for multi-turn conversations, intent classification, and device-specific troubleshooting.

## 🎯 Features

### Core Capabilities
- **Retrieval-Augmented Generation (RAG)**: Answers questions from a knowledge base using semantic search
- **Reasoning Loop & Clarification**: Asks follow-up questions when needed to provide better answers
- **Intent Classification**: Automatically detects user intent (troubleshooting, warranty, cleaning, etc.)
- **Multi-Turn Conversations**: Maintains context across conversation turns
- **Device-Specific Support**: Provides personalized guidance based on FOREO device type
- **Regional Context**: Handles country-specific queries for warranty and orders
- **Paraphrasing**: Uses Gemma-3-270M for light text polishing

### Intent Categories
- **Troubleshooting**: Device issues, charging problems, power issues
- **Cleaning & Care**: Device cleaning instructions
- **Warranty**: Warranty inquiries with regional support
- **Orders & Shipping**: Order tracking, delivery information
- **Account Help**: Account creation, password management
- **Product Inquiry**: Product comparisons and recommendations
- **General QA**: Other FOREO-related questions

## 🏗️ Architecture

### Components

1. **`app.py`**: Main Streamlit application
   - Chat UI with FOREO branding
   - Reasoning loop orchestration
   - Session state management
   - Integration of all components

2. **`rag_gemma.py`**: RAG pipeline core
   - ChromaDB vector store connection
   - Semantic search and retrieval
   - Extractive answer generation
   - Answer paraphrasing with Gemma-3-270M
   - Off-topic detection

3. **`intent_detection.py`**: Intent classification and slot extraction
   - 8 intent categories with confidence scoring
   - Device type extraction (LUNA, BEAR, UFO, ISSA, etc.)
   - Country/region extraction
   - Clarification question generation

4. **`troubleshooting.py`**: Troubleshooting response templates
   - Device-specific troubleshooting steps
   - Common issue resolution guides
   - Formatted response generation

5. **Data Processing**:
   - `prepare_for_embedding.py`: Prepares FAQ data for vector indexing
   - `create_vectorstore.py`: Creates ChromaDB vector store
   - `clean_faqs.py`: Cleans and validates FAQ data

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "FOREO AI"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the data**
```bash
# Clean FAQ data (if needed)
python3 clean_faqs.py

# Prepare data for embedding
python3 prepare_for_embedding.py

# Create vector store
python3 create_vectorstore.py
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🔧 Configuration

### Environment Variables
- `FOREO_LOGO_URL`: Optional URL for FOREO logo image

### File Paths
- `assets/foreo_logo.png`: Logo file (optional)
- `data/cleaned_faqs.jsonl`: Cleaned FAQ data
- `data/faqs_for_embedding.jsonl`: Processed data for embedding
- `chroma_db/`: ChromaDB vector store directory

### Key Parameters (in `rag_gemma.py`)
- `OFFTOPIC_SIM_THRESHOLD`: 0.32 (cosine similarity threshold for off-topic detection)
- `TOP_K`: 3 (number of documents to retrieve)
- `MAX_NEW_TOKENS`: 140 (max tokens for paraphrasing)

## 💬 Usage Examples

### Example 1: Direct Answer (No Clarification)
```
User: "how can I create my account?"
Bot: [Provides complete account creation steps]
```

### Example 2: Device-Specific Clarification
```
User: "how can I clean my device?"
Bot: "To help you better, Which FOREO device would you like cleaning instructions for?"
User: "LUNA 4"
Bot: [Provides LUNA 4 cleaning instructions]
```

### Example 3: Troubleshooting with Clarification
```
User: "my device won't charge"
Bot: "To help you better, Which FOREO device are you having issues with? (e.g., LUNA 4, BEAR, UFO)"
User: "issa mini"
Bot: [Provides troubleshooting steps for ISSA mini charging issues]
```

### Example 4: Regional Clarification
```
User: "what is the warranty period?"
Bot: "Which country are you located in?"
User: "Spain"
Bot: [Provides warranty information for Spain]
```

### Example 5: Complete Information Provided
```
User: "how can I clean my LUNA 4?"
Bot: [Immediately provides LUNA 4 cleaning instructions]
```

## 🔄 Sprint 4: Reasoning Loop Implementation

### Features Added

1. **Intent Detection**
   - Keyword-based classification for 8 intent types
   - Confidence scoring for each intent
   - Proper routing to appropriate response pipeline

2. **Slot Extraction**
   - Device type extraction (LUNA, BEAR, UFO, ISSA, etc.)
   - Country/region extraction (20+ countries supported)
   - Issue type extraction (charging, power, cleaning, etc.)

3. **Clarification System**
   - Asks for missing information only when needed
   - Device-specific: cleaning, troubleshooting, charging
   - Region-specific: warranty, orders
   - Supports natural language responses ("I am in Spain", "spain", "Spain")

4. **Multi-Turn Context**
   - Session state persistence across turns
   - Context merging when user provides clarification
   - Handles short answers like "spain", "luna", "EU"
   - Reconstructs queries from intent when needed

5. **Troubleshooting Flows**
   - Predefined step-by-step guides for common issues
   - Device-specific responses
   - Multiple issue types: charging, power, cleaning, buttons, performance

6. **Session Management**
   - Maintains clarification context across turns
   - Clears context after successful response
   - Handles topic changes gracefully

### Clarification Logic

**Device-Specific (requires device type):**
- Cleaning queries
- Troubleshooting queries
- Charging-related queries

**Region-Specific (requires country):**
- Warranty inquiries
- Order/shipping inquiries

**No Clarification Needed:**
- Account help
- Product inquiries
- General questions

### Supported Countries
United States, United Kingdom, Canada, Australia, Sweden, Germany, France, Italy, Spain, Netherlands, Belgium, Switzerland, Austria, Japan, China, South Korea, Singapore, Turkey, Mexico, Brazil, Argentina, India, Indonesia, Philippines, Thailand, Malaysia, Vietnam, Poland, Portugal, Romania, Hungary, Czech Republic, Slovakia, Greece, Ireland, Denmark, Norway, Finland, Iceland, Russia, Kazakhstan, Uzbekistan, and more.

## 🧪 Testing

### Sprint 4 Test Cases

1. **Cleaning Clarification**
   - Query: "how can I clean my device?"
   - Expected: Asks for device type, then provides cleaning steps

2. **Troubleshooting Clarification**
   - Query: "my device won't charge"
   - Expected: Asks for device type, then provides troubleshooting steps

3. **Regional Clarification**
   - Query: "what is the warranty period?"
   - Expected: Asks for country, then provides country-specific warranty info

4. **No Clarification Needed**
   - Query: "how can I create my account?"
   - Expected: Direct answer without clarification

5. **Complete Query**
   - Query: "how can I clean my LUNA 4?"
   - Expected: Direct answer (device already specified)

## 🛠️ Technical Details

### Intent Classification Algorithm
- Keyword matching with confidence scoring
- Priority-based intent selection
- Falls back to general_qa if no matches

### Slot Extraction
- Regex-based device type detection
- Fuzzy country name matching
- Issue keyword matching for troubleshooting

### Answer Extraction
- Extracts answer from best-matching FAQ document
- Sentence-aware splitting and selection
- Deduplication of repeated content
- Validation of answer quality (filters placeholders)

### Answer Generation Pipeline
1. User query → Intent classification
2. Slot extraction from query
3. Check if clarification needed
4. If yes → Ask clarification question, store context
5. If no → Generate answer:
   - Troubleshooting intent → Use troubleshooting flows
   - Other intents → Use RAG pipeline
6. Optional: Light paraphrasing with Gemma-3-270M
7. Return formatted answer

### Quality Controls
- **Placeholder Filtering**: Removes "A: >" type placeholders
- **Duplicate Detection**: Removes repeated sentences
- **Corruption Detection**: Catches repetitive model outputs
- **Answer Validation**: Ensures minimum meaningful content

## 📊 Performance

- **Retrieval Speed**: ~100-300ms per query
- **Intent Classification**: <10ms
- **Answer Generation**: 200-500ms (including paraphrasing)
- **Supported Devices**: CPU, CUDA, MPS (Apple Silicon)

## 🐛 Troubleshooting

### Common Issues

1. **ChromaDB not found**
   - Run `python3 create_vectorstore.py` to create the vector store

2. **Gemma model fails to load**
   - Check internet connection (first download)
   - Verify sufficient disk space
   - Application will continue without paraphrasing if model fails

3. **Empty answers**
   - Check that FAQ data is properly formatted
   - Verify vector store has indexed documents
   - Check similarity threshold settings

4. **Repetitive responses**
   - Corrupted output detection is in place
   - System automatically falls back to original extracted answer

## 📁 Project Structure

```
FOREO AI/
├── app.py                      # Main Streamlit application
├── rag_gemma.py               # RAG pipeline core
├── intent_detection.py        # Intent classification & slot extraction
├── troubleshooting.py         # Troubleshooting response templates
├── prepare_for_embedding.py   # Data preparation
├── create_vectorstore.py      # Vector store creation
├── clean_faqs.py              # FAQ data cleaning
├── evaluate_rag.py            # Evaluation script
├── requirements.txt          # Python dependencies
├── README.md                  # This file
├── data/
│   ├── cleaned_faqs.jsonl    # Cleaned FAQ data
│   └── faqs_for_embedding.jsonl # Processed data
├── chroma_db/                 # ChromaDB vector store
└── assets/
    └── foreo_logo.png        # Logo image
```

## 🔐 Dependencies

Key packages:
- `streamlit`: Web interface
- `sentence-transformers`: Embeddings (all-MiniLM-L6-v2)
- `chromadb`: Vector database
- `transformers`: Gemma-3-270M model
- `torch`: PyTorch for model inference

See `requirements.txt` for complete list.

## 🚀 Deployment

The application can be deployed using:
- **Streamlit Cloud**: Direct deployment from Git repository
- **Docker**: Containerize the application
- **Local Server**: Run with `streamlit run app.py`

For production deployment, consider:
- Setting up proper error logging
- Monitoring retrieval performance
- Scaling ChromaDB for larger datasets
- Caching model loading for faster startup

## 📝 Development

### Adding New Intents
Edit `intent_detection.py`:
1. Add keywords to `INTENT_KEYWORDS`
2. Define slot schema
3. Add clarification prompts if needed

### Adding New Troubleshooting Flows
Edit `troubleshooting.py`:
1. Add issue type to `TROUBLESHOOTING_FLOWS`
2. Define step-by-step instructions
3. Add device-specific context if applicable

### Extending Country Support
Edit `intent_detection.py` in `extract_country()`:
1. Add country name variations to the `countries` dictionary
2. Support will automatically extend to warranty/orders queries

## 📄 License

[Specify your license here]

## 👥 Authors

Team 9 - Botirjon

## 🎉 Sprint 4 Status: ✅ COMPLETE

All Sprint 4 acceptance criteria have been met:
- ✅ Intent detection with >90% accuracy
- ✅ Slot schema storage in session state
- ✅ Clarifying questions (only when needed)
- ✅ Troubleshooting flows with step-by-step guides
- ✅ Integration with RAG pipeline
- ✅ Session memory across conversation turns
- ✅ Ready for evaluation (>85% accuracy target)
