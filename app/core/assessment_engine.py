"""
Scientific Hypertrophy Trainer - Assessment Engine (FIXED)
Simplified logic to prevent question skipping.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

class AssessmentEngine:
    def __init__(self, questions_file="data/questions.json"):
        self.questions_file = questions_file
        self.questions_data = None
        self.current_assessment = None
        self.current_tier = None
        self.current_question_index = 0
        self.shuffled_questions = []
        self.answers = []
        
        self.load_questions()
        
    def load_questions(self):
        try:
            questions_path = Path(self.questions_file)
            with open(questions_path, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            print(f"✅ Loaded {self.get_total_question_count()} questions")
        except Exception as e:
            print(f"❌ Error loading questions: {e}")
            self.questions_data = {}
    
    def get_total_question_count(self) -> int:
        if not self.questions_data: return 0
        total = 0
        for tier in self.questions_data.values():
            total += len(tier.get('questions', []))
        return total
    
    def start_assessment(self, tier_level: int) -> Dict:
        tier_keys = ['tier_1_fundamentals', 'tier_2_intermediate', 'tier_3_advanced']
        if not 0 <= tier_level < len(tier_keys): return {}
        
        tier_data = self.questions_data.get(tier_keys[tier_level])
        
        # Reset State
        self.current_tier = tier_level
        self.current_assessment = tier_data
        self.current_question_index = 0
        self.answers = []
        
        # Shuffle a fresh copy
        self.shuffled_questions = tier_data['questions'].copy()
        random.shuffle(self.shuffled_questions)
        
        print(f"🎯 Started Tier {tier_level + 1}. Loaded {len(self.shuffled_questions)} questions.")
        
        return {
            'tier_level': tier_level,
            'total_questions': len(self.shuffled_questions),
            'passing_percentage': tier_data['passing_percentage'],
            'passing_score': int(len(self.shuffled_questions) * tier_data['passing_percentage'])
        }
    
    def get_current_question(self) -> Optional[Dict]:
        """Simple, linear retrieval. No complex recursion."""
        if not self.shuffled_questions or self.current_question_index >= len(self.shuffled_questions):
            return None
        
        question = self.shuffled_questions[self.current_question_index]
        
        # Return with meta-data
        q_copy = question.copy()
        q_copy['question_number'] = self.current_question_index + 1
        q_copy['total_questions'] = len(self.shuffled_questions)
        
        return q_copy
    
    def submit_answer(self, selected_option_text: str) -> Dict:
        current_q = self.shuffled_questions[self.current_question_index]
        
        # Check correctness
        correct_opt = next((o for o in current_q['options'] if o['correct']), None)
        is_correct = correct_opt and selected_option_text == correct_opt['text']
        
        # Record
        self.answers.append({
            'question_id': current_q['id'],
            'question_text': current_q['question'],
            'selected_answer': selected_option_text,
            'correct_answer': correct_opt['text'] if correct_opt else "Error",
            'is_correct': is_correct,
            'explanation': current_q.get('explanation', '')
        })
        
        # Advance
        self.current_question_index += 1
        
        if self.current_question_index >= len(self.shuffled_questions):
            return {'status': 'complete', 'results': self.finish_assessment()}
        else:
            return {'status': 'next'}
    
    def finish_assessment(self) -> Dict:
        total = len(self.shuffled_questions)
        correct = sum(1 for a in self.answers if a['is_correct'])
        pct = (correct / total * 100) if total > 0 else 0
        pass_req = self.current_assessment['passing_percentage'] * total
        passed = correct >= pass_req
        
        print(f"📊 Assessment Complete: {correct}/{total} ({pct:.1f}%)")
        
        return {
            'tier_level': self.current_tier,
            'score': correct,
            'total_questions': total,
            'percentage': pct,
            'passed': passed,
            'answers': self.answers,
            'next_tier_unlocked': passed and self.current_tier < 2
        }