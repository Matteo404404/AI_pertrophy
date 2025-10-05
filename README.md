# 🏋️ Scientific Hypertrophy Trainer

> **Evidence-based muscle building through progressive knowledge assessment and intelligent data analytics**

An advanced desktop application that transforms hypertrophy training from guesswork into precision science. Built with Python, PyQt6, and machine learning, this system guides lifters from fundamentals to expert-level programming through a comprehensive knowledge tier system, granular tracking, and (coming soon) adaptive ML recommendations.

---

## 🎯 Project Vision

This isn't just another fitness tracker—it's a **complete hypertrophy intelligence platform** designed to demonstrate advanced software engineering, data science capabilities, and deep domain expertise in exercise science.

**The Goal:** Build the most sophisticated training companion that combines:
- Rigorous sports science principles
- Production-grade architecture
- Machine learning optimization engines
- Intuitive, professional UI/UX

Even in this **alpha stage**, the codebase showcases enterprise-level patterns: modular core engines, normalized database schema, comprehensive error handling, and a polished interface that rivals commercial applications.

---

## ✨ Current Features (Alpha v0.1)

### 🎨 **Modern UI/UX**
- Clean, iOS-inspired light theme across all interfaces
- Smooth animations and responsive design
- Professional typography and spacing
- Intuitive navigation with sidebar menu

### 📊 **Intelligent Dashboard**
- Real-time progress metrics
- Current tier status and completion rates
- Training consistency indicators
- Quick-access navigation cards

### 🧠 **Tiered Knowledge Assessment System**
- **60 peer-reviewed questions** across 3 progressive tiers
- **Tier 1 (Fundamentals):** Volume, intensity, frequency, nutrition basics
- **Tier 2 (Intermediate):** Effective reps, muscle length relationships, periodization
- **Tier 3 (Advanced):** MRV estimation, autoregulation, regional hypertrophy optimization

**Features:**
- Adaptive question randomization (no repeats)
- 80% passing threshold with instant scoring
- Automatic tier unlocking on completion
- Progress bars and real-time feedback
- Detailed explanations for every question

### 📅 **Comprehensive Tracking System**
**Calendar-Based Data Entry:**
- Diet tracking: macros, micros, hydration, meal timing, supplementation
- Sleep analysis: duration, quality, REM/deep sleep estimation
- Body measurements: weight, body fat %, circumferences (8+ sites)
- Workout logging: exercises, sets, reps, load, RIR/RPE

**Technical Highlights:**
- Date-specific data persistence (entries don't bleed between days)
- SQLite backend with proper indexing and relationships
- Real-time form validation
- Smart defaults based on user profile

### 📚 **Adaptive Learning Center**
- Categorizes all missed assessment questions by concept
- Groups errors into actionable study areas:
  - Training Volume
  - Training Intensity  
  - Training Frequency
  - Exercise Selection
  - Nutrition & Recovery
  - General Principles

**Features:**
- Shows your incorrect answer vs. correct answer
- Detailed evidence-based explanations
- Priority ranking by mistake frequency
- Study progress tracking

### 👥 **Multi-User System**
- Beautiful user selection screen
- Individual profiles with independent progression
- Quick user switching
- User-specific data isolation
- Import/export capabilities (planned)

---

## 🚀 Upcoming Features (In Development)

### **Phase 1: Machine Learning Integration** 🤖
Currently implementing advanced ML models to provide personalized recommendations:

#### **1. Progress Prediction Engine**
- Time-series forecasting for muscle growth and strength gains
- ARIMA/Prophet models with confidence intervals
- Accounts for volume, frequency, nutrition, sleep quality
- Visual prediction charts with historical overlay

#### **2. Personalized Volume Optimization**
- Random Forest regression to find individual MEV (Minimum Effective Volume)
- Dynamic MRV (Maximum Recoverable Volume) estimation
- Per-muscle-group volume recommendations
- Adaptive scaling based on recovery metrics

#### **3. Recovery & Readiness Scoring**
- Multi-factor ML classification model
- Weighted algorithm: sleep (40%), nutrition (30%), training load (30%)
- Daily readiness recommendations: "Train Hard", "Moderate", "Deload", "Rest"
- Pattern recognition for overtraining prevention

#### **4. Exercise Selection Recommender**
- Collaborative filtering algorithm (similar to Netflix recommendations)
- Analyzes muscle imbalances, equipment availability, injury history
- Progressive exercise substitution suggestions
- Difficulty scaling based on technique mastery

### **Phase 2: Advanced Analytics** 📈
- Per-muscle volume/fatigue tracking dashboards
- Stimulus-to-fatigue ratio analysis
- Periodization block planning
- Progressive overload trend visualization
- Correlation analysis (sleep quality → performance, etc.)

### **Phase 3: Polish & Scale** ✨
- Cloud sync for multi-device access
- Mobile companion app (React Native)
- Social features (anonymous progress comparison)
- Export comprehensive PDF reports
- Exercise video library integration
- Refined UI animations and micro-interactions

---

## 🏗️ Technical Architecture

### **Technology Stack**
```
Frontend:     PyQt6 (Desktop GUI)
Backend:      Python 3.10+
Database:     SQLite3 with normalized schema
ML/Analytics: scikit-learn, pandas, numpy
Visualization: matplotlib, seaborn (planned)
Testing:      pytest, unittest (planned)
```

### **Project Structure**
```
scientific-hypertrophy-trainer/
├── gui/                      # PyQt6 interface modules
│   ├── main_window.py       # Main application window & navigation
│   ├── user_selection.py    # User management interface
│   ├── dashboard.py         # Progress dashboard
│   ├── assessment.py        # Knowledge assessment UI
│   ├── tracking.py          # Diet/sleep/workout logging
│   └── learning.py          # Learning center interface
│
├── core/                     # Domain logic & business rules
│   ├── assessment_engine.py # Question loading, scoring, tier progression
│   ├── tracking_system.py   # Analytics engine for training data
│   └── user_manager.py      # User authentication & profile management
│
├── database/                 # Data layer
│   └── db_manager.py        # SQLite interface with comprehensive schema
│
├── ml/                       # Machine learning models (in development)
│   ├── progress_predictor.py
│   ├── volume_optimizer.py
│   └── recovery_scorer.py
│
├── data/                     # Application data
│   ├── questions.json       # 60 validated hypertrophy questions
│   └── users.db            # SQLite database
│
├── resources/               # Assets
│   ├── icons/
│   └── styles/
│
├── tests/                   # Unit & integration tests (planned)
│
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### **Database Schema Highlights**
- **Users:** Profile data, tier progression, preferences
- **Assessments:** Completed exams, scores, timestamps
- **Assessment Answers:** Question-level detail for learning center
- **Diet Entries:** Comprehensive macro/micronutrient tracking
- **Sleep Entries:** Multi-dimensional sleep quality metrics
- **Body Measurements:** Time-series anthropometric data
- **Workout Logs:** Exercise-level training data (sets, reps, load, RPE)

*All tables properly indexed with foreign key constraints and cascade deletes.*

---

## 🎓 Scientific Foundation

Every feature is built on peer-reviewed research from:
- Dr. Brad Schoenfeld (hypertrophy mechanisms)
- Dr. Mike Israetel (volume landmarks, MEV/MRV/MAV concepts)  
- Dr. Eric Helms (training pyramid, evidence hierarchy)
- Renaissance Periodization (periodization, exercise selection)
- Greg Nuckols (programming principles, biomechanics)

**60 Assessment Questions** cover:
- Mechanical tension as primary growth stimulus
- Volume dosing strategies (MEV → MAV → MRV)
- Effective reps and proximity to failure
- Frequency optimization (2-3x per muscle per week)
- Exercise selection criteria (length-tension relationships)
- Nutrition timing and total protein intake
- Recovery optimization (sleep, deloads, autoregulation)

---

## 🚀 Getting Started

### **Prerequisites**
- Python 3.10 or higher
- pip package manager
- 100MB disk space

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/scientific-hypertrophy-trainer.git
cd scientific-hypertrophy-trainer
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python main.py
```

### **First Launch**
1. Create your first user profile
2. Complete the Tier 1 assessment (16/20 to pass)
3. Start tracking workouts, diet, and sleep
4. Review mistakes in the Learning Center
5. Watch your dashboard populate with insights!

---

## 📊 Development Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| **Alpha v0.1** | ✅ Complete | Core UI, Assessment System, Tracking, Learning Center |
| **Alpha v0.2** | 🔄 In Progress | ML Progress Predictor, Volume Optimizer |
| **Beta v0.3** | 📋 Planned | Recovery Scoring, Exercise Recommender |
| **Beta v0.4** | 📋 Planned | Advanced Analytics, Visualization Suite |
| **v1.0** | 🎯 Future | Cloud Sync, Mobile App, Community Features |

---

## 🤝 Contributing

This is a personal portfolio project, but suggestions and feedback are welcome!

**Areas where contributions would be valuable:**
- Additional peer-reviewed questions for Tier 2/3
- Bug reports with reproduction steps
- UI/UX improvement suggestions
- Performance optimization ideas
- ML model architecture recommendations

**Contribution Guidelines:**
- Follow PEP 8 style guidelines
- Include comprehensive docstrings with type hints
- Provide scientific citations for training-related changes
- Write clear commit messages
- Test thoroughly before submitting PR

---

## 📝 Technical Highlights for Recruiters

This project demonstrates:

### **Software Engineering**
✅ **Clean Architecture:** Separation of concerns (GUI, core logic, data layer)  
✅ **Design Patterns:** Factory, Observer, Singleton, Strategy patterns  
✅ **Error Handling:** Comprehensive try-catch with user-friendly messaging  
✅ **Database Design:** Normalized schema with proper indexing  
✅ **Code Quality:** Type hints, docstrings, consistent naming conventions  

### **Data Science (In Development)**
✅ **ML Pipeline:** Data preprocessing, feature engineering, model training  
✅ **Time Series:** ARIMA/Prophet forecasting for progress prediction  
✅ **Regression:** Volume optimization using Random Forest/Gradient Boosting  
✅ **Classification:** Recovery readiness scoring with multi-factor analysis  
✅ **Recommender Systems:** Collaborative filtering for exercise selection  

### **Domain Expertise**
✅ **Exercise Science:** Deep understanding of hypertrophy mechanisms  
✅ **Evidence Synthesis:** 60+ questions validated against research  
✅ **Practical Application:** Real-world training considerations  

### **UI/UX Design**
✅ **Modern Aesthetics:** Clean, professional interface  
✅ **User Research:** Intuitive flows based on lifter needs  
✅ **Consistency:** Unified design system across all screens  

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Matteo Melis**  
Data Science Student | TU/e & Tilburg University  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

Built with insights from the hypertrophy research community:
- Renaissance Periodization
- Stronger By Science  
- Dr. Brad Schoenfeld's research group
- 3DMJ (Eric Helms, Andrea Valdez, Alberto Nunez)

*All training recommendations in this application are based on peer-reviewed scientific literature and expert consensus. Always consult qualified professionals before beginning any training program.*

---

**⭐ Star this repo if you find it useful!**

*Last Updated: October 2025*
