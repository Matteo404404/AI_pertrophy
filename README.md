This is a **God-Tier README**. It is designed to look professional on GitHub, explain the complex scientific architecture of your project, and serve as a complete manual for any athlete or developer.

I have selected the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license, which allows anyone to use and remix the code for free, but legally forbids them from selling it or making money from it.

***

# 🧬 AI_Pertrophy: Scientific Hypertrophy Trainer
> **The world’s first SOTA Hypertrophy Training App powered by Biomathematical Modeling, LSTM Neural Networks, and RAG-based AI Coaching.**

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-orange.svg)
![AI](https://img.shields.io/badge/AI-Ollama%20%7C%20LSTM-green.svg)

---

## 🏛 Project Mission
Most fitness apps are built on "bro-science" or generic templates. **AI_Pertrophy** is built on peer-reviewed exercise physiology. It ignores trends like "stretch-mediated hypertrophy" hype and focuses on the primary driver of muscle growth: **High-Force Mechanical Tension** and **Motor Unit Recruitment**. 

The app uses an **LSTM (Long Short-Term Memory) Neural Network** and the **Banister Fitness-Fatigue Model** to predict your strength levels and manage your fatigue so you never hit a plateau.

---

## 🚀 God-Tier Features
*   **🧠 Hybrid AI Brain:** Routes between a Banister-Heuristic model (for new users) and a deep-learning LSTM model (for experienced users).
*   **💬 RAG AI Coach:** A local LLM (`qwen3:1.7b`) that reads a scientific hypertrophy database to answer your specific training questions.
*   **📈 Interactive Analytics HUD:** Hardware-accelerated charts showing SRA (Stimulus-Recovery-Adaptation) curves, 1RM trends, and Muscle Readiness heatmaps.
*   **🧪 Nutrition HUD:** Live-link to the USDA Foundation database. Fetch 14+ micronutrients (Zinc, Magnesium, Omega-3s) via fuzzy string matching.
*   **🏆 Knowledge Assessment:** A 3-Tier clearance system that unlocks app features only once you prove you understand training principles.

---

## 📂 System Architecture (The File Map)

### 1. `app/` (The Application Layer)
*   **`core/`**: The "Logic Center."
    *   `assessment_engine.py`: Manages quiz logic and tier unlocking.
    *   `nutrition_lookup.py`: The SOTA engine that talks to USDA API and Ollama for food parsing.
    *   `user_manager.py`: Handles athlete profiles and biometrics.
*   **`database/`**:
    *   `db_manager.py`: The SQLite backbone. Manages 15+ tables including exercise performance and micro-nutrition history.
*   **`gui/`**: The "HUD."
    *   `main_window.py`: The primary navigation shell and sidebar.
    *   `analytics.py`: Real-time data processing for charts.
    *   `tracking.py`: The "Console" where you log sessions and food.
    *   `food_search_dialog.py`: Modern search interface with live macro previews.

### 2. `ml_engine/` (The Mathematical Brain)
*   `models/pytorch_strength_predictor.py`: The LSTM Neural Network architecture with Multi-Head Attention.
*   `models/simple_lifting_predictor.py`: The Banister Fitness-Fatigue mathematical model.
*   `inference/hybrid_predictor.py`: The intelligent router that decides which AI model to use for your data.

### 3. `research_lab/` (The Simulation Center)
*   `generator/enhanced_synthetic_generator.py`: A physiological simulator that generates millions of data points based on the **Henneman Size Principle** to train the AI.

---

## 🛠 Installation & Setup

### Prerequisites
1.  **Python 3.14+**
2.  **Ollama**: [Download here](https://ollama.com/).
3.  **USDA API Key**: [Get your free key here](https://fdc.nal.usda.gov/api-key-signup.html).

### Steps
1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/yourusername/AI_pertrophy.git
    cd AI_pertrophy
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment:**
    Create a `.env` file in the root folder:
    ```text
    USDA_API_KEY=your_key_here
    ```
4.  **Download the AI Model:**
    ```bash
    ollama pull qwen3:1.7b
    ```
5.  **Run the App:**
    ```bash
    python main.py
    ```

---

## 📖 User Manual: The Athlete Lifecycle

### 1. The Clearance Phase (Assessment)
You begin as a "Tier 0" user. Go to the **Assessment** tab and pass the **Tier 1: Fundamentals** exam. If you fail, the **AI Coach** will appear to explain your mistakes based on scientific literature.

### 2. The Loading Phase (Tracking)
*   **Workout:** Go to **Tracking Logs**, select a Scientific Protocol (e.g., "Upper A — Max Force"), and click **Inject Protocol**. Enter your Load (kg) and RIR (Reps in Reserve).
*   **Nutrition:** Click **Search USDA Database**. Search for a food, select your unit (grams, oz, cups), and add it to your log.

### 3. The Analysis Phase (Analytics)
Visit the **Analytics** tab to see:
*   **SRA Trajectory:** Are you recovered? If the Blue Readiness line is below 0, **do not train today.**
*   **SFR Matrix:** Look for dots in the "Optimal" quadrant. Move dots out of the "Junk Volume" quadrant.
*   **Weekly Debrief:** Click the button at the bottom to have the AI write a full report on your week.

---

## 🔬 The Science (Training Philosophy)
This app is built on the following pillars:
1.  **Henneman's Size Principle:** Motor units are recruited based on force demand. We prioritize 0-2 RIR to ensure high-threshold motor unit recruitment.
2.  **Diminishing Returns of Volume:** Based on **Schoenfeld (2017)**, volume is logarithmic. The app warns you if you exceed 22 sets per session (Junk Volume).
3.  **Banister Model:** Performance = Fitness - Fatigue. We track both integrals to predict your 1RM accurately.
4.  **Anti-Fragility:** We avoid "stretch-mediated" focus in high-damage ranges to prioritize frequency and force production over unnecessary soreness (DOMS).

---

## 📜 License
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

*   **Attribution:** You must give appropriate credit to the original creator.
*   **Non-Commercial:** You **MAY NOT** use this material for commercial purposes (selling the app, charging for access, or using it as a paid service).
*   **Share Alike:** If you remix or build upon the material, you must distribute your contributions under the same license.

---

