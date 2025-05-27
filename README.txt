# EKoderLocal 🧠

**EKoderLocal** is a privacy-first, AI-powered diagnosis coding tool for Australian Emergency Departments.

It runs entirely **offline on your Mac**, using a local Large Language Model (LLM) to suggest accurate ICD-10-AM principal diagnosis codes based on ED case notes.

---

## 🚀 Quick Start (Mac)

### 1. Clone or download the app

From Terminal:

```bash
git clone https://github.com/Diogenes67/EKoderLocal.git
cd EKoderLocal

Or click Code → Download ZIP, then unzip the folder.
2. Install dependencies (one time)

Make sure Python is installed, then run:

pip install streamlit sentence-transformers pandas openpyxl plotly

Also install Ollama, and run this once:

ollama run llama3:8b-instruct-q4_K_M

3. Launch the app

From Terminal:

./launch_ekoder.command

This will open the app in your browser at http://localhost:8501
📦 What’s Included
File	Description
ekoderlocal.py	Streamlit app source code
FinalEDCodes_Complexity.xlsx	ED ICD-10-AM shortlist and funding scale
edcode_finetune_v5_updated.jsonl	Few-shot examples for prompting
ekoderlocal_styles.css	Optional UI custom styles
logo.png	App logo
launch_ekoder.command	🔁 Double-click Mac launcher
README.md	This file
🔒 Privacy

    No cloud services used

    No patient data leaves your machine

    Entirely local model inference via Ollama and open-source models

    Suitable for clinical research, hospital R&D, and governance-safe deployments

🧠 What it Does

    Accepts a case note pasted or uploaded

    Uses local embedding + LLM to suggest 1–4 mutually exclusive ICD-10-AM codes

    Returns:

        Code

        Description

        AI rationale

        Complexity (🟣–🔴) for funding insight

🧰 Developer Tools

Want to rebuild the launcher as a native binary?

pyinstaller --onefile --windowed ekoder_launcher.py

👨‍⚕️ Created by @Diogenes67

Built by a clinician–engineer for real-world Emergency Departments.
Feedback, issues, and collaboration welcome!


---

## ✅ How to add it

If you're ready to add:

```bash
cd ~/Desktop/EKoderLocal
nano README.md

Paste the full text above, then:

    Ctrl + O, Enter to save

    Ctrl + X to exit

    Then commit:

git add README.md
git commit -m "Add full project README"
git push

