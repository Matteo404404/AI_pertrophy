# AI_Pertrophy — Hypertrophy Training App

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyQt6-orange.svg)

A desktop training app built around evidence-based hypertrophy principles. It uses an LSTM neural network and the Banister fitness-fatigue model to predict strength progression and manage training fatigue.

## What it does

- **Hybrid AI predictor** — routes between a Banister heuristic (new users) and a trained LSTM model (enough data)
- **Local AI coach** — a RAG chatbot powered by Ollama (`qwen3:1.7b`) that answers training questions using a built-in knowledge base
- **Analytics** — SRA curves, 1RM trends, muscle readiness heatmaps, weekly AI debrief
- **Nutrition tracking** — USDA FoodData Central API integration with micronutrient support
- **Knowledge assessment** — 3-tier clearance system that tests understanding of training principles before unlocking features
- **76 exercises** with biomechanical metadata (resistance profile, stability score, regional bias)

## Project structure

```
app/
  core/          — assessment engine, nutrition lookup, user manager, tracking system
  database/      — SQLite manager (15+ tables)
  gui/           — PyQt6 interface (dashboard, tracking, analytics, learning, assessment)
ml_engine/
  models/        — LSTM predictor, Banister heuristic, collaborative filtering
  inference/     — hybrid predictor, preprocessor
research_lab/
  generator/     — synthetic data generator (Henneman size principle, Banister model)
  training/      — PyTorch training pipeline, dataset
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` in the project root:**
   ```
   USDA_API_KEY=your_key_here
   ```

3. **Pull the AI model (optional, for the chatbot):**
   ```bash
   ollama pull qwen3:1.7b
   ```

4. **Run:**
   ```bash
   python main.py
   ```

## Training the ML model

```bash
python train_model.py --num_users 5000 --num_epochs 50
```

This generates synthetic training data, trains the LSTM, and saves the model to `ml_engine/models/`.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You can use and remix this for non-commercial purposes with attribution. You may not sell it or use it as a paid service.
