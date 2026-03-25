"""
Scientific Hypertrophy Trainer - Learning Center v4.0
Research-backed knowledge base drawing from:
- Schoenfeld (2010, 2017, 2021) - mechanisms, volume dose-response
- Krieger (2010) - set-volume meta-analysis
- Morton et al. (2018) - protein meta-analysis
- Damas et al. (2015) - muscle damage vs hypertrophy
- Wernbom et al. (2007) - frequency and rep ranges
- Beardsley (2019, 2020) - length-tension, resistance profiles
- McDonald (2009) - rate of muscle gain models
- Helms et al. (2014, 2015) - RIR/RPE, evidence-based recommendations
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QScrollArea, QTabWidget, QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt

KNOWLEDGE_DB = {
    "Mechanical Tension (Primary Driver)": {
        "Mechanical Tension = Force Production": (
            "Mechanical tension is force produced by cross-bridge cycling under load. "
            "The INTENT to move the weight explosively is what maximally recruits motor "
            "units — not deliberately slowing the rep down. When the load is heavy or "
            "you're fatigued, the bar moves slowly involuntarily despite maximal effort. "
            "That involuntary slowdown under maximal intent is what produces the highest "
            "tension. A 'grinder' rep at RIR 0 where you're pushing as hard as possible "
            "produces dramatically more tension than an easy rep at RIR 5. Always try to "
            "accelerate the concentric — the load or fatigue will take care of the speed."
        ),
        "High-Threshold Motor Unit Recruitment": (
            "Per Henneman's Size Principle (1965), motor units are recruited from "
            "smallest to largest as force demands increase. The largest motor units "
            "(Type II fibers — the ones with the highest hypertrophic potential) are "
            "only fully recruited under high force demands: either heavy loads (>80% 1RM) "
            "or lighter loads taken very close to failure. If you stop a set at 3+ RIR "
            "with moderate loads, you never fully recruit the fibers that matter most."
        ),
        "Why Muscle Damage Is NOT a Driver": (
            "Damas et al. (2015) showed that muscle damage peaks in the first ~3 sessions "
            "of a novel stimulus, then drops to near-zero — yet hypertrophy continues "
            "linearly for months. This proves that damage is a byproduct of novelty, "
            "not a cause of growth. Chasing soreness (DOMS) wastes recovery resources. "
            "If you're constantly sore, you're either switching exercises too often or "
            "exceeding your MRV. Neither is productive."
        ),
        "Metabolic Stress: Overstated": (
            "Cell swelling ('the pump'), lactate accumulation, and reactive oxygen species "
            "were once proposed as independent growth drivers. Schoenfeld (2013) initially "
            "gave this weight, but subsequent research (e.g., BFR studies, Lasevicius 2018) "
            "shows that when mechanical tension is equated, metabolic stress adds little or "
            "nothing. The pump feels good; it does not cause meaningful additional growth."
        ),
    },
    "Volume, Intensity & Dose-Response": {
        "Effective Reps — The Only Reps That Count": (
            "Not all reps in a set are hypertrophic. Only the last ~5 reps before "
            "muscular failure fully recruit high-threshold motor units AND impose "
            "sufficient tension duration. These are 'effective reps' or 'stimulating reps'. "
            "A set of 20 at RIR 0 has roughly the same number of effective reps as a "
            "set of 8 at RIR 0 — but the set of 20 generates far more cardiovascular "
            "and metabolic fatigue. This is why moderate rep ranges (5-12) dominate "
            "practical recommendations: they maximize effective reps per unit of fatigue."
        ),
        "Volume: Logarithmic, Not Linear": (
            "Schoenfeld et al. (2017) meta-analysis showed a dose-response curve for "
            "weekly sets per muscle group. But critically, the curve is LOGARITHMIC: "
            "going from 1→5 sets adds far more growth than going from 10→15 sets. "
            "Krieger (2010) found that ~2-3 hard sets per exercise per session captures "
            "the majority of the stimulus. Doing 5+ sets of the same exercise in one "
            "session usually just accumulates fatigue (Junk Volume) without proportional "
            "stimulus. Weekly volume of 10-20 sets per muscle group is the practical "
            "sweet spot for most intermediates — but the QUALITY of those sets matters "
            "more than the quantity."
        ),
        "RIR-Based Autoregulation (Helms et al. 2016)": (
            "RIR (Reps In Reserve) is a validated method for regulating intensity. "
            "0 RIR = failure; 1 RIR = could do 1 more rep. Research-backed targets: "
            "0-2 RIR for isolation exercises (low systemic cost). 1-3 RIR for compounds "
            "(higher injury/fatigue risk at 0 RIR). Beginners chronically underestimate "
            "their RIR — what they call '2 RIR' is often actually 4-5 RIR. A key skill "
            "is learning to accurately assess your RIR, which improves with experience."
        ),
        "MEV, MAV, MRV (Israetel Framework)": (
            "MEV (Minimum Effective Volume): The fewest sets that produce measurable "
            "growth. Often ~6 sets/muscle/week. MAV (Maximum Adaptive Volume): Where "
            "most people grow best. Typically 12-20 sets/week. MRV (Maximum Recoverable "
            "Volume): Exceeding this causes regression. Highly individual — depends on "
            "sleep, nutrition, stress, training age. The key insight: MRV is NOT fixed. "
            "It changes week to week based on recovery status. Autoregulation > rigid plans."
        ),
    },
    "Biomechanics & Exercise Selection": {
        "Resistance Profiles — Matching the Strength Curve": (
            "Every exercise has a 'resistance profile' — where in the ROM the external "
            "load is hardest. Every muscle has a 'strength curve' — where it produces "
            "the most force. The goal is to choose exercises where the resistance is "
            "highest WHERE THE MUSCLE IS STRONGEST, so you can produce maximal force "
            "through the full ROM. Cables provide constant tension throughout. Machines "
            "can be engineered to match specific curves. Free weights have fixed "
            "ascending/descending profiles dictated by gravity and lever arms. Exercise "
            "selection should prioritize the ability to produce HIGH FORCE safely — "
            "not chase any particular muscle length."
        ),
        "The Stretch-Mediated Hype — What the Research Actually Shows": (
            "Studies like Pedrosa (2023) and Maeo (2021) comparing long-length partials "
            "to short-length partials found more growth in the long-length group. But "
            "this is often misinterpreted. These studies compared PARTIALS — not full "
            "ROM. Full ROM training already captures whatever benefit exists at long "
            "lengths. The practical takeaway is: use full ROM, not that you should seek "
            "extreme stretch positions. Training at extreme stretch adds disproportionate "
            "muscle DAMAGE (which Damas 2015 showed is NOT a growth driver), increases "
            "injury risk, and impairs recovery — meaning you can train LESS frequently. "
            "The effect also appears muscle-specific — it does not generalize to all "
            "muscles equally. Exercises like the seated leg curl and overhead triceps "
            "extension are good because they avoid active insufficiency and allow higher "
            "FORCE PRODUCTION, not because of 'stretch magic.' Do not design your "
            "training around stretching muscles under load."
        ),
        "Concentric Intent — Explosive Drives Motor Unit Recruitment": (
            "The intent to move the weight as fast as possible during the concentric "
            "phase is what maximally recruits high-threshold motor units (Behm & Sale "
            "1993). Deliberately slowing the concentric reduces force production, which "
            "LOWERS motor unit recruitment — the opposite of what you want. When the "
            "load is heavy enough, the bar will move slowly regardless of intent. That "
            "combination of maximal effort + slow movement is where tension peaks. "
            "Slow eccentrics (3-4 seconds) have a place for control and feel, but "
            "excessively slow tempos on the concentric are counterproductive. Push hard, "
            "control the eccentric, repeat."
        ),
        "Stability and Motor Unit Recruitment": (
            "When external stability is high (machines, chest-supported), the CNS can "
            "direct 100% of neural drive to the prime movers. When stability is low "
            "(standing overhead press, single-leg work), significant neural resources "
            "go to stabilizer muscles and balance. For HYPERTROPHY (not athletic performance), "
            "machines and supported movements produce higher per-muscle stimulus with lower "
            "systemic fatigue. Free weights are not inherently 'better' — they just cost more "
            "recovery per unit of muscle stimulus."
        ),
        "Active Insufficiency — The Real Reason Exercise Variants Matter": (
            "A biarticular muscle cannot produce maximal force when shortened across both "
            "joints simultaneously. This is why exercise variants exist — not for 'stretch' "
            "but for FORCE PRODUCTION. Example: the hamstrings are both a knee flexor and "
            "hip extensor. A lying leg curl with hips extended puts the hamstrings in active "
            "insufficiency — shortened at the hip, trying to shorten at the knee. The muscle "
            "literally cannot contract as hard. A SEATED leg curl (hips flexed) avoids this, "
            "allowing the hamstrings to produce MORE FORCE per rep. Same logic: overhead "
            "triceps extension lets the long head produce more force than pushdowns. This is "
            "about biomechanical advantage, not about stretching anything."
        ),
    },
    "Recovery & Nutrition": {
        "Protein: The 1.6 g/kg Threshold (Morton et al. 2018)": (
            "The largest protein-dose meta-analysis found that protein intake above "
            "~1.6 g/kg/day has minimal additional hypertrophic benefit. The range "
            "1.6-2.2 g/kg captures the confidence interval — going to 3+ g/kg does "
            "not hurt, but the extra protein contributes nearly zero additional MPS. "
            "Distribution matters less than total daily intake, though spreading "
            "protein across 3-5 meals of ≥0.3 g/kg each may be slightly optimal "
            "(Schoenfeld & Aragon 2018). A 80kg person should target 128-176g/day."
        ),
        "Sleep Architecture & Growth Hormone": (
            "Deep sleep (SWS) is when pulsatile GH release peaks, supporting tissue "
            "repair. REM sleep restores neural drive, which directly impacts next-day "
            "training quality. Knowles et al. (2018) found that chronic sleep restriction "
            "(<6h) reduces anabolic hormone profiles and increases catabolic markers. "
            "Practical: 7-9 hours is the minimum for serious trainees. Sleep quality "
            "(latency, awakenings, environment) matters as much as duration. A bad 9 hours "
            "is worse than a good 7."
        ),
        "Caloric Surplus & Rate of Gain (McDonald)": (
            "Lyle McDonald's models estimate realistic muscle gain rates: ~1-1.5% of BW/month "
            "for beginners (year 1), dropping to ~0.25% by year 4+. A moderate surplus of "
            "200-500 kcal/day above maintenance supports maximal muscle gain with minimal "
            "fat accumulation. Larger surpluses do NOT accelerate muscle growth — they only "
            "add fat faster. For recomposition (gaining muscle while losing fat), a slight "
            "deficit or maintenance is possible for beginners/detrained individuals, but "
            "advanced lifters need a surplus."
        ),
        "Creatine: The Only Legal Supplement That Works": (
            "Creatine monohydrate is the most extensively studied ergogenic supplement. "
            "It saturates intramuscular phosphocreatine stores, improving ATP regeneration "
            "during high-intensity efforts (sets of 1-15 reps). Meta-analyses show a "
            "consistent ~5-8% improvement in strength performance and lean mass accrual. "
            "5g/day is sufficient; loading phases are unnecessary (they just reach "
            "saturation faster). Timing does not matter. Every other supplement has "
            "dramatically weaker evidence (caffeine for performance, vitamin D if deficient, "
            "fish oil for general health)."
        ),
    },
    "Periodization & Programming": {
        "Why Periodization Exists": (
            "Periodization is NOT about 'confusing' the muscle. It exists because "
            "recovery capacity is limited and the stimulus-to-fatigue ratio degrades "
            "as volume accumulates across a mesocycle. A typical accumulation block "
            "(4-6 weeks) progressively increases volume from MEV toward MRV. When "
            "performance begins to stagnate or decline, a deload (reduced volume to ~50%) "
            "dissipates accumulated fatigue while retaining fitness. This is the "
            "Fitness-Fatigue model (Banister 1976): performance = fitness - fatigue."
        ),
        "Mesocycle Structure": (
            "A well-designed mesocycle for hypertrophy: Week 1 at MEV (e.g., 10 sets/muscle), "
            "progressing by 1-2 sets/week until reaching MRV by Week 4-5 (e.g., 18-22 sets). "
            "Then deload in Week 5-6 (drop to ~6 sets, keep intensity). The next meso starts "
            "at a slightly higher MEV than the last (progressive overload at the macro level). "
            "This is more effective than running maximal volume every week, which leads to "
            "accumulated fatigue and overreaching within 3-4 weeks."
        ),
        "Resensitization & Strategic Deloads": (
            "After prolonged high-volume training, muscle fibers become desensitized to the "
            "anabolic signaling cascade (mTOR pathway). A strategic maintenance phase "
            "(2-3 weeks at MEV with high intensity) can restore this sensitivity, making the "
            "next accumulation block more productive per set. This is why advanced lifters "
            "who 'do less' for a planned period often grow MORE in the subsequent block than "
            "those who just grind volume year-round."
        ),
        "Frequency: 2x/week Per Muscle (Schoenfeld 2016)": (
            "Schoenfeld et al. (2016) meta-analysis found that training each muscle ≥2x/week "
            "produces superior hypertrophy compared to 1x/week (bro splits), likely because "
            "it spikes MPS more frequently and allows volume to be distributed across sessions "
            "(reducing per-session fatigue). However, 3-4x/week does not clearly beat 2x/week "
            "for most people. The practical recommendation: Upper/Lower or Push/Pull/Legs splits "
            "hitting each muscle 2-3x/week, distributing weekly volume across those sessions."
        ),
    },
    "Common Misconceptions": {
        "'More Volume Is Always Better'": (
            "This is the single most damaging myth in training. The dose-response curve for "
            "volume has a clear ceiling (MRV), beyond which you LOSE muscle. The SRA curve "
            "(Stimulus-Recovery-Adaptation) requires complete recovery before the next "
            "overloading session. Excessive volume delays recovery, shifts the SRA curve "
            "rightward, and produces a state of functional overreaching. Most naturals are "
            "better served doing 2-3 hard sets per exercise than 5-6 moderate sets."
        ),
        "'Muscle Confusion' & Exercise Rotation": (
            "Changing exercises frequently does NOT produce superior hypertrophy. It DOES "
            "produce more DOMS (via novel stimuli), which people misinterpret as 'the "
            "workout is working.' In reality, constantly rotating exercises prevents you "
            "from progressively overloading any single movement, and the repeated-bout "
            "effect (Nosaka 2008) means adaptation to a movement REDUCES damage over time — "
            "which is desirable, not something to fight against."
        ),
        "'Light Weight, High Reps for Toning'": (
            "There is no physiological distinction between 'toning' and hypertrophy. "
            "Muscle can grow or shrink; it cannot change shape independent of size. "
            "The appearance of 'toned' muscles is simply hypertrophy + low body fat. "
            "High-rep training (15-30) CAN build muscle if taken to failure, but it "
            "produces disproportionate cardiovascular fatigue and discomfort relative to "
            "the mechanical tension achieved. It is not more 'toning' — it is just a less "
            "efficient stimulus pathway for most people."
        ),
        "'You Need to Feel the Burn'": (
            "The 'burn' is hydrogen ion accumulation from anaerobic glycolysis. It correlates "
            "with metabolic stress, NOT mechanical tension. An exercise can produce intense burn "
            "with minimal hypertrophic stimulus (e.g., wall sits, high-rep bodyweight squats) "
            "or no burn with enormous stimulus (e.g., a heavy set of 5 on RDLs). Using the "
            "burn as a proxy for stimulus quality leads to program designs that maximize "
            "discomfort while minimizing actual growth signal."
        ),
    },
}


class LearningWidget(QWidget):
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Knowledge Base & Education")
        header.setStyleSheet("font-size: 28px; font-weight: 800; color: #89b4fa;")
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_skill_tab(), "My Skill Profile")
        self.tabs.addTab(self.create_library_tab(), "Concept Library")
        self.tabs.addTab(self.create_mistakes_tab(), "Mistake Review")

        main_layout.addWidget(self.tabs)

    def create_skill_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        info = QLabel("AI confidence scores based on your assessment history.")
        info.setStyleSheet("color: #a6adc8; font-style: italic; margin-bottom: 20px;")
        layout.addWidget(info)

        self.skill_container = QFrame()
        self.skill_container.setStyleSheet(
            "background-color: #262639; border-radius: 12px; padding: 20px;")
        self.skill_layout = QVBoxLayout(self.skill_container)
        self.skill_layout.setSpacing(20)

        layout.addWidget(self.skill_container)
        layout.addStretch()
        return tab

    def create_library_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 20, 0, 0)

        topic_frame = QFrame()
        topic_frame.setFixedWidth(280)
        topic_frame.setStyleSheet("background: #181825; border-radius: 8px;")
        t_layout = QVBoxLayout(topic_frame)

        self.topic_list = QListWidget()
        self.topic_list.setStyleSheet(
            "border: none; background: transparent; color: #cdd6f4;")
        for category in KNOWLEDGE_DB.keys():
            self.topic_list.addItem(category)
        self.topic_list.currentItemChanged.connect(self.load_topic_content)

        t_layout.addWidget(self.topic_list)
        layout.addWidget(topic_frame)

        content_frame = QFrame()
        content_frame.setStyleSheet("background: #1e1e2e;")
        c_layout = QVBoxLayout(content_frame)

        self.lbl_topic = QLabel("Select Topic")
        self.lbl_topic.setStyleSheet(
            "font-size: 24px; color: #fab387; font-weight: bold;")
        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setStyleSheet(
            "background: #262639; border-radius: 8px; padding: 15px; "
            "font-size: 14px; color: #cdd6f4; line-height: 1.6;")

        c_layout.addWidget(self.lbl_topic)
        c_layout.addWidget(self.txt_content)
        layout.addWidget(content_frame)
        return tab

    def create_mistakes_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.mistakes_area = QScrollArea()
        self.mistakes_area.setWidgetResizable(True)
        self.mistakes_area.setStyleSheet("border: none; background: transparent;")

        self.mistakes_container = QWidget()
        self.mistakes_layout = QVBoxLayout(self.mistakes_container)
        self.mistakes_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.mistakes_area.setWidget(self.mistakes_container)
        layout.addWidget(self.mistakes_area)
        return tab

    def refresh_data(self):
        self.load_skills()
        self.load_mistakes()

    def load_skills(self):
        while self.skill_layout.count():
            child = self.skill_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        user = self.user_manager.get_current_user()
        if not user:
            return

        profile = self.db.get_user_ml_profile(user['id'])
        skills = [
            ("Training Literacy", profile.get('training_literacy_index', 0.5)),
            ("Recovery IQ", profile.get('recovery_knowledge', 0.5)),
            ("Load Management", profile.get('load_management_score', 0.5)),
            ("Technique", profile.get('technique_score', 0.5)),
        ]

        for name, score in skills:
            self.add_skill_bar(name, score)

    def add_skill_bar(self, name, score):
        pct = int(score * 100)
        lbl = QLabel(f"{name}: {pct}%")
        lbl.setStyleSheet("color: white; font-weight: bold;")
        bar = QProgressBar()
        bar.setValue(pct)
        if pct > 70:
            color = '#a6e3a1'
        elif pct > 40:
            color = '#f9e2af'
        else:
            color = '#f38ba8'
        bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}")
        self.skill_layout.addWidget(lbl)
        self.skill_layout.addWidget(bar)

    def load_topic_content(self, item):
        if not item:
            return
        cat = item.text()
        self.lbl_topic.setText(cat)
        html = ""
        for term, desc in KNOWLEDGE_DB.get(cat, {}).items():
            html += f"<h3 style='color: #89b4fa; margin-top: 20px;'>{term}</h3>"
            html += f"<p style='line-height: 1.7;'>{desc}</p>"
        self.txt_content.setHtml(html)

    def load_mistakes(self):
        while self.mistakes_layout.count():
            child = self.mistakes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        user = self.user_manager.get_current_user()
        if not user:
            return

        assessments = self.db.get_user_assessments(user['id'])
        found = False
        for a in assessments:
            answers = self.db.get_assessment_answers(a['id'])
            for ans in answers:
                if not ans['is_correct']:
                    self.add_mistake_card(ans)
                    found = True

        if not found:
            self.mistakes_layout.addWidget(QLabel("No mistakes found. Good job!"))

    def add_mistake_card(self, answer_data):
        card = QFrame()
        card.setStyleSheet(
            "background: #262639; border-radius: 8px; border-left: 4px solid #f38ba8;")
        layout = QVBoxLayout(card)

        q_text = answer_data.get('question_text') or f"Question ID: {answer_data.get('question_id')}"
        layout.addWidget(QLabel(f"<b>Question:</b> {q_text}"))

        hbox = QHBoxLayout()
        wrong = QLabel(f"Your answer: {answer_data['user_answer']}")
        wrong.setStyleSheet("color: #f38ba8;")
        right = QLabel(f"Correct: {answer_data['correct_answer']}")
        right.setStyleSheet("color: #a6e3a1;")

        hbox.addWidget(wrong)
        hbox.addWidget(right)
        layout.addLayout(hbox)

        self.mistakes_layout.addWidget(card)
