🧠 EKoderLocal – Easy Instructions (Mac)

=======================================
STEP 1: Unzip the File
=======================================
1. Download EKoderLocal.zip
2. Double-click it to unzip
✅ A folder named EKoderLocal will appear

=======================================
STEP 2: Run the App
=======================================
1. Open the EKoderLocal folder
2. Find the file called: launch_ekoder.command
3. Double-click it
✅ This opens the app in your browser at: http://localhost:8501

=======================================
STEP 3: First-Time Setup (only once)
=======================================

❓ If the app doesn't open, follow these steps:

▶️ How to open Terminal:
- Click the magnifying glass 🔍 at the top-right (Spotlight)
- Type: Terminal
- Press Enter

▶️ Then copy and paste this into Terminal:

pip install streamlit sentence-transformers pandas openpyxl plotly

▶️ Then double-click launch_ekoder.command again.

=======================================
STEP 4: One-Time Ollama Setup
=======================================
1. Install Ollama from https://ollama.com
2. Then in Terminal, run:

ollama run llama3:8b-instruct-q4_K_M

You only need to do this once.

=======================================
If you’re stuck
=======================================
Try launching from Terminal directly:

cd ~/Downloads/EKoderLocal
./launch_ekoder.command

=======================================
You’re all set!
=======================================
Once setup is done, you can launch EKoderLocal anytime by double-clicking the file.

