import streamlit as st
from PIL import Image
from pathlib import Path
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
import plotly.graph_objects as go
import requests
import warnings
import sys

# Fix: Suppress Streamlit cache warnings  
warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")


# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="EKoderLocal – Privacy-First ED Code Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Ollama Integration Functions ===
def check_ollama_status():
    """Check if Ollama is running and Llama model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            return 'llama3:8b-instruct-q4_K_M' in available_models
        return False
    except:
        return False

def predict_final_codes_local(note, shortlist_df, fewshot, model="llama3:8b-instruct-q4_K_M"):
    """Local LLM processing - replaces OpenAI GPT-4o."""
    options_text = "\n".join(
        f"{r['ED Short List code']} — {r['ED Short List Term']}" for _, r in shortlist_df.iterrows()
    )
    
    prompt = f"""You are the head of the emergency department and an expert clinical coder.
Your rationale should help other doctors understand the pros and cons of choosing each code.

Your task is to suggest between **one and four mutually exclusive ED Short List ICD-10-AM codes** that could each plausibly serve as the **principal diagnosis**, based on the diagnostic content of the casenote.

These codes are **not** intended to build a combined clinical picture — rather, they should reflect **alternative coding options**, depending on the coder's interpretation and emphasis. **Each code must stand on its own** as a valid representation of the case presentation.

---

**How to think it through (show your work):**
1. Identify the single finding or cluster of findings that most tightly matches one code.
2. Pick that code as **#1 (best fit)** and provide a clear justification:
   - Show exactly which language in the note drives your choice
   - Highlight why it's more specific or higher-priority than the next option
3. Repeat for up to **4 total**, each time choosing the next-best fit.
4. If no highly specific match remains, choose the least-specific fallback — but **do not** use R69.
5. **Do not** list comorbidities or incidental findings unless they truly dominate the presentation.

---

**ED Code Shortlist:**
{options_text}

**Casenote:**
{note}

---

**Output Format (exactly):**
1. CODE — "<your rationale>"
2. CODE — "<your rationale>"
3. … up to 4

Please follow that structure precisely.
"""
    
    try:
        payload = {
            "model": model,
            "prompt": fewshot + prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1000}
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            st.error(f"Local LLM error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error calling local LLM: {e}")
        return None

def local_css(file_name):
    try:
        print("📄 Trying to load CSS from:", file_name)
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")
    except Exception as e:
        st.error(f"Unexpected error loading CSS: {e}")

# Load EKoderLocal styles

# Load logo
try:
    logo = Image.open("/Users/anthonyporter/Desktop/Coding/EKlogo.png")
    st.image(logo, width=150)
except Exception as e:
    st.warning(f"Logo not loaded: {e}")

# Title and description  
st.markdown("<h1 style='color:#007AC1;font-size:48px;'>🔒 EKoderLocal</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:18px; color:#333;'>
<b>EKoderLocal</b> is a privacy-first clinical coding tool for Australian Emergency Departments.<br><br>

<b>🔒 Complete Privacy:</b> All AI processing happens locally on your device using open-source models. 
Case notes never leave your laptop - no cloud APIs, no external servers, no data transmission.<br><br>

This tool analyzes free-text ED case notes and suggests <b>up to four ICD-10-AM principal diagnosis codes</b> 
using local Large Language Models (Llama 3.1) for complete data privacy and compliance.<br><br>

<b>How it works:</b><br>
EKoderLocal uses local semantic similarity to identify relevant codes, then processes them with 
a privacy-preserving local LLM for reasoning and ranking. Perfect for sensitive healthcare environments.<br><br>

⚠️ <b>Please ensure all case notes are de-identified before processing.</b><br>
<b>For research and educational purposes only.</b> Not intended for patient care or formal documentation.<br><br>

<b>Privacy-first healthcare AI by Amplar.</b>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = None
if 'note_text' not in st.session_state:
    st.session_state.note_text = ""
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# === Configuration Variables ===
# Local LLM status check
LOCAL_LLM_AVAILABLE = check_ollama_status()

# Force local embeddings for privacy
USE_LOCAL_EMBEDDINGS = True
st.session_state.embedding_mode = "Local"

# File path configurations
st.sidebar.header("🔒 EKoderLocal Status")

if LOCAL_LLM_AVAILABLE:
    st.sidebar.success("✅ Local LLM Ready")
    st.sidebar.info("🛡️ Privacy-preserving processing enabled")
else:
    st.sidebar.error("❌ Local LLM not available")
    st.sidebar.markdown("""
    **To enable EKoderLocal:**
    ```bash
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Download Llama 3 instruct model
    ollama pull llama3:8b-instruct-q4_K_M
    ```
    """)

st.sidebar.info("🔒 **Privacy Mode**: Using local embeddings only")
# Hardcoded Root Directory
ROOT = Path("/Users/anthonyporter/edcode_streamlit/amplicodr/amplicodr/amplicodr-desktop")
local_css(str(ROOT / "ekoderlocal_styles.css"))

print("📂 ROOT directory set to:", ROOT)
print("📊 Checking default Excel file:", (ROOT / "FinalEDCodes_Complexity.xlsx").exists())
print("📊 Checking default JSONL file:", (ROOT / "edcode_finetune_v5_updated.jsonl").exists())

# File paths with file uploaders
st.sidebar.subheader("Required Files")

# For Excel file
DEFAULT_EXCEL = ROOT / "FinalEDCodes_Complexity.xlsx"
uploaded_excel = st.sidebar.file_uploader("Upload ICD Codes Excel", type=["xlsx", "xls"], help="Optional if default file is present")

if uploaded_excel:
    EXCEL_PATH = ROOT / uploaded_excel.name
    with open(EXCEL_PATH, "wb") as f:
        f.write(uploaded_excel.getbuffer())
    st.sidebar.success(f"✅ Uploaded Excel: {uploaded_excel.name}")
elif DEFAULT_EXCEL.exists():
    EXCEL_PATH = DEFAULT_EXCEL
    st.sidebar.success(f"✅ Using default: FinalEDCodes_Complexity.xlsx")
else:
    EXCEL_PATH = None
    st.sidebar.error("❌ No Excel file found. Please upload one.")


# For JSONL file
DEFAULT_JSONL = ROOT / "edcode_finetune_v5_updated.jsonl"
uploaded_jsonl = st.sidebar.file_uploader("Upload Few-Shot Examples", type=["jsonl"], help="Optional if default file is present")

if uploaded_jsonl:
    JSONL_PATH = ROOT / uploaded_jsonl.name
    with open(JSONL_PATH, "wb") as f:
        f.write(uploaded_jsonl.getbuffer())
    st.sidebar.success(f"✅ Uploaded JSONL: {uploaded_jsonl.name}")
elif DEFAULT_JSONL.exists():
    JSONL_PATH = DEFAULT_JSONL
    st.sidebar.success("✅ Using default: edcode_finetune_v5_updated.jsonl")
else:
    JSONL_PATH = None
    st.sidebar.error("❌ No JSONL file found. Please upload one.")

# Path for embedding cache
EMBEDDING_CACHE_PATH = ROOT / "ed_code_embeddings.pkl"

# Model definitions - using better local models
LOCAL_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LOCAL_DIM = 768

# Emoji lookup for funding scale visualization
funding_emojis = {
    1: "🟣", 2: "🔵", 3: "🟢",
    4: "🟡", 5: "🟠", 6: "🔴"
}

# === Utility Functions ===
def cosine(u, v):
    """Return cosine similarity between vectors u and v."""
    return np.dot(u, v) / (norm(u) * norm(v))

@st.cache_data
def get_embeddings_local(texts):
    """Obtain embeddings using local SentenceTransformer model."""
    texts = list(texts)  # convert Series to list safely
    print("📦 get_embeddings_local() called")
    print("🧪 Number of texts to embed:", len(texts))
    print("🔍 First input sample:", texts[0][:80] if len(texts) > 0 else "EMPTY")

    try:
        from sentence_transformers import SentenceTransformer
        print("🤖 Loading SentenceTransformer model...")
        model = SentenceTransformer(LOCAL_MODEL_NAME, device="cpu")
        print("🤖 Model loaded. Embedding now...")

        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("✅ Embeddings generated:", len(embeddings))
        if len(embeddings) > 0:
            print("🧬 Shape of first embedding:", embeddings[0].shape)
        return embeddings

    except ImportError:
        st.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Error with local embeddings: {e}")
        print("❌ Embedding error:", e)
        return None

@st.cache_data
def build_code_embeddings(descriptions, cache_path, use_local=True):
    """Build or load cached embeddings for the code descriptions."""
    expected_dim = LOCAL_DIM
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                embeds = pickle.load(f)
            dim = embeds.shape[1] if isinstance(embeds, np.ndarray) and embeds.ndim == 2 else (
                len(embeds[0]) if isinstance(embeds, list) and len(embeds) > 0 else None
            )
            if dim == expected_dim:
                st.sidebar.success(f"Loaded {len(embeds)} cached embeddings ({dim}d)")
                return embeds
            else:
                st.sidebar.info(f"Cache dimension {dim} != expected {expected_dim}; regenerating...")
        except Exception as e:
            st.sidebar.warning(f"Error loading cache: {e}")

    # Generate embeddings
    with st.sidebar.status("Generating local embeddings..."):
        embeds = get_embeddings_local(descriptions)

        if embeds is not None and len(embeds) > 0:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(embeds, f)
                st.sidebar.success(f"Generated and cached {len(embeds)} embeddings")
            except Exception as e:
                st.sidebar.warning(f"Failed to cache embeddings: {e}")
        else:
            st.sidebar.error("Failed to generate embeddings")
            return None

    return embeds

def get_funding_emoji(code, funding_lookup):
    """Return emoji representing funding scale for a given code."""
    return funding_emojis.get(funding_lookup.get(code, 3), "🟩")

@st.cache_data
def get_top_matches(note_emb, code_embs, df, top_n=5):
    """Compute cosine similarity between note embedding and each code embedding."""
    sims = [cosine(note_emb, e) for e in code_embs]
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    top = df.iloc[idx].copy()
    top['Similarity'] = [sims[i] for i in idx]
    return top

@st.cache_data
def load_examples(path, limit=3):
    """Load few-shot examples from a .jsonl file for prompt context."""
    path = Path(path)
    if not path.exists():
        st.error(f"Example file not found: {path}")
        return ""

    ex = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit: break
                d = json.loads(line)
                ex.append(
                    f"Casenote:\n{d['messages'][0]['content']}\nAnswer:\n{d['messages'][1]['content']}"
                )
        return "\n\n---\n\n".join(ex) + "\n\n---\n\n"
    except Exception as e:
        st.error(f"Error loading examples: {e}")
        return ""

def parse_response(resp, df):
    """Parse the LLM response to extract ICD codes and explanations."""
    valid = set(df['ED Short List code'].astype(str).str.strip())
    term = dict(zip(df['ED Short List code'], df['ED Short List Term']))
    funding_lookup = dict(zip(df['ED Short List code'], df['Scale'].fillna(3).astype(int)))
    
    rows = []
    for line in resp.splitlines():
        # Updated regex to handle Llama 3 format: 1. CODE — Description — "rationale"
        m = re.match(r"\d+\.\s*([A-Z0-9\.]+)\s*[—-]\s*\"?([^\"]*)\"?", line)
        if m:
            code, expl = m.groups()
            if code in valid and code != 'R69':
                rows.append((code, term[code], expl.strip('"').strip("'"), get_funding_emoji(code, funding_lookup)))
    return rows

def process_batch_files_local(uploaded_files, excel_path, jsonl_path, embedding_cache_path):
    """Process multiple uploaded files with local LLM."""
    results = []
    
    try:
        # Load Excel data
        raw = pd.read_excel(excel_path)
        raw.columns = raw.columns.str.strip() 
        raw = raw.rename(columns={
            "ED Short": "ED Short List code",
            "Diagnosis": "ED Short List Term", 
            "Descriptor": "ED Short List Included conditions"
        })
        
        desc_list = (raw["ED Short List Term"] + ". " + raw["ED Short List Included conditions"].fillna(""))
        funding_lookup = dict(zip(raw["ED Short List code"], raw["Scale"].fillna(3).astype(int)))
        
        # Use local embeddings only
        code_embeddings = build_code_embeddings(desc_list, embedding_cache_path, use_local=True)
        fewshot = load_examples(jsonl_path)
        
        # Process each file with local LLM
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            content = uploaded_file.getvalue().decode("utf-8")
            
            try:
                # Local embeddings for the note
                note_emb = get_embeddings_local([content])[0]
                shortlist = get_top_matches(note_emb, code_embeddings, raw, 12)
                
                # Use local LLM
                resp = predict_final_codes_local(content, shortlist, fewshot)
                print(f"DEBUG - Raw LLM Response: {resp}")
                print(f"DEBUG - Raw response for {filename}: {resp[:500] if resp else None}...")
                parsed = parse_response(resp, raw)
                
                # Format results
                codes_with_complexity = []
                for code, term, explanation, emoji in parsed:
                    complexity = funding_lookup.get(code, 3)
                    codes_with_complexity.append({
                        'code': code,
                        'term': term,
                        'explanation': explanation,
                        'complexity': complexity,
                        'emoji': emoji
                    })
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'error': None,
                    'codes': codes_with_complexity
                })
            
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e),
                    'codes': []
                })
                
    except Exception as e:
        return [{"filename": "system_error", "success": False, "error": f"System error: {str(e)}", "codes": []}]
    
    return results

# === Main Application Logic ===

# Add tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Batch Processing"])

with tab1:
    st.header("Enter Case Note")
    note_text = st.text_area("Type or paste the case note here:", height=300, 
                            value=st.session_state.note_text)

with tab2:
    st.header("Upload Case Note")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    if uploaded_file:
        note_text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("File contents:", note_text, height=300)

with tab3:
    st.header("Batch Processing")
    st.markdown("Upload multiple case notes for bulk processing")
    
    uploaded_files = st.file_uploader(
        "Choose multiple text files", 
        type=["txt"], 
        accept_multiple_files=True,
        key="batch_files"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} files for processing")
        for i, f in enumerate(uploaded_files, 1):
            st.write(f"{i}. {f.name}")
        
        if st.button("Process All Files", type="primary", key="batch_process"):
            if not LOCAL_LLM_AVAILABLE:
                st.error("Local LLM is required for batch processing")
            else:
                with st.spinner("Processing files with local LLM... This may take a few minutes"):
                    batch_results = process_batch_files_local(
                        uploaded_files, 
                        EXCEL_PATH,
                        JSONL_PATH,
                        EMBEDDING_CACHE_PATH
                    )
                    st.session_state.batch_results = batch_results
                
                st.success(f"Completed processing {len(uploaded_files)} files!")

# Save note text to session state
if note_text:
    st.session_state.note_text = note_text

# Configure top_n parameter
top_n = st.sidebar.slider("Number of similar codes to consider", min_value=5, max_value=20, value=12)

# Single file processing button
if st.button("Classify Note", type="primary", disabled=not bool(note_text and LOCAL_LLM_AVAILABLE), key="classify_single"):
    if not EXCEL_PATH.exists():
        st.error(f"Excel file not found at {EXCEL_PATH}")
    elif not JSONL_PATH.exists():
        st.error(f"JSONL file not found at {JSONL_PATH}")
    else:
        with st.spinner("Loading data and building embeddings..."):
            # Load the Excel data
            raw = pd.read_excel(EXCEL_PATH)
            raw.columns = raw.columns.str.strip()
            raw = raw.rename(columns={
                "ED Short": "ED Short List code",
                "Diagnosis": "ED Short List Term",
                "Descriptor": "ED Short List Included conditions"
            })
            desc_list = (raw["ED Short List Term"] + ". " + raw["ED Short List Included conditions"].fillna(""))
            
            # Build embeddings (local only)
            code_embeddings = build_code_embeddings(desc_list, EMBEDDING_CACHE_PATH, use_local=True)
            if code_embeddings is None:
                st.error("Failed to build embeddings. Please check your configuration.")
                st.stop()

            # Load few-shot examples
            fewshot = load_examples(JSONL_PATH)

        with st.spinner("Computing embeddings for note..."):
            # Get local embeddings for the note
            note_emb = get_embeddings_local([note_text])[0]

            if note_emb is None:
                st.error("Failed to generate embeddings for the note.")
                st.stop()

            # Get top similar codes
            shortlist = get_top_matches(note_emb, code_embeddings, raw, top_n)

        with st.spinner("Consulting local LLM for diagnosis..."):
            # Query local LLM
            resp = predict_final_codes_local(note_text, shortlist, fewshot)
            print(f"DEBUG - Raw LLM Response: {resp}")            
            if resp is None:
                st.error("Failed to get response from local LLM.")
                st.stop()

            # Parse LLM response
            parsed = parse_response(resp, raw)

            # Save results to session state
            st.session_state.results = {
                "shortlist": shortlist,
                "gpt_response": resp,
                "parsed_results": parsed
            }

# Display single file results if available
if st.session_state.results:
    # Display shortlist
    with st.expander("Embedding Shortlist", expanded=False):
        st.dataframe(
            st.session_state.results["shortlist"][["ED Short List code", "ED Short List Term", "Similarity"]],
            use_container_width=True,
            hide_index=True
        )

    # Display Local LLM raw response
    with st.expander("Local LLM Raw Response", expanded=False):
        st.code(st.session_state.results["gpt_response"])

    # Display final results
    st.header("Classification Results")
    if st.session_state.results["parsed_results"]:
        results_df = pd.DataFrame(
            st.session_state.results["parsed_results"], 
            columns=["Code", "Term", "Explanation", "Emoji"]
        )

        # Create a Plotly table with Amplar branding
        fig = go.Figure(data=[go.Table(
           columnwidth=[60, 180, 600, 80],
           header=dict(
                values=["Code", "Term", "Explanation", "Complexity"],
                fill_color='rgb(0, 122, 193)',  # Amplar blue
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[
                    results_df["Code"],
                    results_df["Term"],
                    results_df["Explanation"],
                    results_df["Emoji"]
                ],
                fill_color='rgb(248, 248, 248)',
                align='left',
                font=dict(size=13),
                height=40
            )
        )])

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=125 * len(results_df)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No valid codes extracted from the response.")

# Display batch results if available
if st.session_state.batch_results:
    st.header("Batch Processing Results")
    
    # Build summary table
    summary_data = []
    for result in st.session_state.batch_results:
        row = {
            'Filename': result['filename'],
            'Status': 'SUCCESS' if result['success'] else 'ERROR',
            'Codes Found': len(result['codes']) if result['success'] else 0,
        }
        # Add first 4 codes and complexity
        for i in range(4):
            if result['success'] and i < len(result['codes']):
                code_data = result['codes'][i]
                row[f'Code {i+1}'] = code_data['code']
                row[f'Scale {i+1}'] = code_data['complexity']
            else:
                row[f'Code {i+1}'] = ""
                row[f'Scale {i+1}'] = ""
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ekoderlocal_batch_results.csv",
        mime="text/csv"  
    )

# Complexity Scale Legend
st.markdown("""
### 🧾 Complexity Scale Legend

The **Complexity** value reflects the typical resource use associated with each diagnosis code in the Emergency Department setting, based on historical funding data.

<table style="width:100%; font-size:16px; border-collapse:collapse;">
  <thead>
    <tr>
      <th align="left">Scale</th>
      <th align="left">Funding Range (AUD)</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>🟣 1</td><td>≤ $499</td><td>Minimal complexity</td></tr>
    <tr><td>🔵 2</td><td>$500 – $699</td><td>Low complexity</td></tr>
    <tr><td>🟢 3</td><td>$700 – $899</td><td>Moderate complexity</td></tr>
    <tr><td>🟡 4</td><td>$900 – $1099</td><td>High complexity</td></tr>
    <tr><td>🟠 5</td><td>$1100 – $1449</td><td>Significant complexity</td></tr>
    <tr><td>🔴 6</td><td>≥ $1450</td><td>Very high complexity</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

# Display instructions in the sidebar
with st.sidebar.expander("Instructions", expanded=False):
    st.markdown("""
    ### How to use EKoderLocal

    1. Ensure Ollama is running with Llama 3 model
    2. Upload the required Excel and JSONL files (optional)
    3. Enter a case note, upload a file, or batch process multiple files
    4. Click 'Classify Note' to get ICD code recommendations

    ### About EKoderLocal

    This application uses:
    - Local Llama 3 instruct model for AI reasoning
    - Local sentence-transformers for semantic similarity
    - Complete privacy - no data leaves your device
    - Streamlit for the web interface
    
    ### Privacy Features
    - 🔒 Local AI processing only
    - 🚫 No external API calls
    - 🛡️ Data never transmitted
    - 🏠 Runs entirely on your device
    """)
