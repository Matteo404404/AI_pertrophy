"""
Scientific Hypertrophy Trainer - Assessment Engine
Manages question loading, randomization, scoring WITHOUT repeats
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class AssessmentEngine:
    """
    Handles all assessment logic:
    - Load questions from JSON
    - Track used questions (NO REPEATS)
    - Randomize question order
    - Score assessments with 80% threshold
    - Generate learning recommendations
    """
    
    def __init__(self, questions_file="data/questions.json"):
        """Initialize assessment engine and load questions"""
        self.questions_file = questions_file
        self.questions_data = None
        self.current_assessment = None
        self.current_tier = None
        self.current_question_index = 0
        self.used_question_ids = set()  # Track used questions
        self.answers = []
        
        self.load_questions()
        
    def load_questions(self):
        """Load questions from JSON file"""
        try:
            questions_path = Path(self.questions_file)
            with open(questions_path, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            print(f"✅ Loaded {self.get_total_question_count()} questions from {self.questions_file}")
        except FileNotFoundError:
            print(f"❌ Questions file not found: {self.questions_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in questions file: {e}")
            raise
    
    def get_total_question_count(self) -> int:
        """Get total number of questions across all tiers"""
        if not self.questions_data:
            return 0
        
        total = 0
        for tier_key in ['tier_1_fundamentals', 'tier_2_intermediate', 'tier_3_advanced']:
            if tier_key in self.questions_data:
                total += len(self.questions_data[tier_key].get('questions', []))
        return total
    
    def get_tier_data(self, tier_level: int) -> Optional[Dict]:
        """Get tier data by level (0=Tier1, 1=Tier2, 2=Tier3)"""
        tier_keys = ['tier_1_fundamentals', 'tier_2_intermediate', 'tier_3_advanced']
        if 0 <= tier_level < len(tier_keys):
            return self.questions_data.get(tier_keys[tier_level])
        return None
    
    def start_assessment(self, tier_level: int) -> Dict:
        """
        Start a new assessment for specified tier
        Returns: Dictionary with assessment info
        """
        tier_data = self.get_tier_data(tier_level)
        if not tier_data:
            raise ValueError(f"Invalid tier level: {tier_level}")
        
        # Reset assessment state
        self.current_tier = tier_level
        self.current_assessment = tier_data
        self.current_question_index = 0
        self.used_question_ids.clear()  # Clear for new assessment
        self.answers = []
        
        # Shuffle questions for randomization (but track IDs to prevent repeats)
        self.shuffled_questions = self.current_assessment['questions'].copy()
        random.shuffle(self.shuffled_questions)
        
        print(f"🎯 Started Tier {tier_level + 1} assessment: {tier_data['title']}")
        print(f"   Questions: {len(self.shuffled_questions)}")
        print(f"   Passing: {int(len(self.shuffled_questions) * tier_data['passing_percentage'])} correct")
        
        return {
            'tier_level': tier_level,
            'tier_title': tier_data['title'],
            'tier_description': tier_data['description'],
            'total_questions': len(self.shuffled_questions),
            'passing_percentage': tier_data['passing_percentage'],
            'passing_score': int(len(self.shuffled_questions) * tier_data['passing_percentage'])
        }
    
    def get_current_question(self) -> Optional[Dict]:
        """
        Get current question (without repeats)
        Returns: Question dictionary or None if assessment complete
        """
        if not self.current_assessment or self.current_question_index >= len(self.shuffled_questions):
            return None
        
        question = self.shuffled_questions[self.current_question_index]
        
        # Check if already used (shouldn't happen with proper flow, but safety check)
        if question['id'] in self.used_question_ids:
            print(f"⚠️ Question {question['id']} already used, skipping...")
            self.current_question_index += 1
            return self.get_current_question()  # Recursively get next
        
        # Mark as used
        self.used_question_ids.add(question['id'])
        
        # Add progress info
        question_with_progress = question.copy()
        question_with_progress['question_number'] = self.current_question_index + 1
        question_with_progress['total_questions'] = len(self.shuffled_questions)
        question_with_progress['progress_percentage'] = round(
            (self.current_question_index + 1) / len(self.shuffled_questions) * 100
        )
        
        return question_with_progress
    
    def submit_answer(self, selected_option_text: str) -> Dict:
        """
        Submit answer for current question
        Args:
            selected_option_text: The text of the selected answer option
        Returns: Dictionary with answer result and next status
        """
        if not self.current_assessment:
            raise ValueError("No assessment in progress")
        
        current_question = self.shuffled_questions[self.current_question_index]
        
        # Find correct answer
        correct_option = next(
            (opt for opt in current_question['options'] if opt['correct']),
            None
        )
        
        if not correct_option:
            raise ValueError(f"Question {current_question['id']} has no correct answer marked")
        
        # Check if answer is correct
        is_correct = selected_option_text == correct_option['text']
        
        # Record answer
        answer_record = {
            'question_id': current_question['id'],
            'question_text': current_question['question'],
            'selected_answer': selected_option_text,
            'correct_answer': correct_option['text'],
            'is_correct': is_correct,
            'explanation': current_question.get('detailed_explanation', '')
        }
        
        self.answers.append(answer_record)
        
        # Move to next question
        self.current_question_index += 1
        
        # Check if assessment is complete
        if self.current_question_index >= len(self.shuffled_questions):
            return {
                'status': 'complete',
                'answer': answer_record,
                'results': self.finish_assessment()
            }
        else:
            return {
                'status': 'continue',
                'answer': answer_record,
                'next_question': self.get_current_question()
            }
    
    def finish_assessment(self) -> Dict:
        """
        Complete assessment and calculate results
        Returns: Dictionary with score and pass/fail status
        """
        if not self.current_assessment:
            raise ValueError("No assessment to finish")
        
        total_questions = len(self.shuffled_questions)
        correct_count = sum(1 for ans in self.answers if ans['is_correct'])
        percentage = round((correct_count / total_questions) * 100, 1)
        
        passing_percentage = self.current_assessment['passing_percentage']
        passing_score = int(total_questions * passing_percentage)
        passed = correct_count >= passing_score
        
        results = {
            'tier_level': self.current_tier,
            'tier_title': self.current_assessment['title'],
            'score': correct_count,
            'total_questions': total_questions,
            'percentage': percentage,
            'passing_score': passing_score,
            'passing_percentage': passing_percentage * 100,
            'passed': passed,
            'answers': self.answers.copy(),
            'next_tier_unlocked': passed and self.current_tier < 2
        }
        
        print(f"📊 Assessment Complete:")
        print(f"   Score: {correct_count}/{total_questions} ({percentage}%)")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return results
    
    def get_wrong_answers(self, answers: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Get list of wrong answers for learning system
        Args:
            answers: List of answer records (uses current if not provided)
        Returns: List of wrong answer details with explanations
        """
        if answers is None:
            answers = self.answers
        
        wrong_answers = [ans for ans in answers if not ans['is_correct']]
        
        # Add concept categorization
        for answer in wrong_answers:
            answer['concept_area'] = self.categorize_question(answer['question_text'])
        
        return wrong_answers
    
    def categorize_question(self, question_text: str) -> str:
        """Categorize question into concept area for learning system"""
        text_lower = question_text.lower()
        
        # Categorization based on keywords
        if any(keyword in text_lower for keyword in ['volume', 'sets', 'mrv', 'mev']):
            return 'Training Volume'
        elif any(keyword in text_lower for keyword in ['intensity', 'failure', 'rir', 'reps in reserve']):
            return 'Training Intensity'
        elif any(keyword in text_lower for keyword in ['protein', 'nutrition', 'carbohydrate', 'diet']):
            return 'Nutrition'
        elif any(keyword in text_lower for keyword in ['sleep', 'recovery', 'rest']):
            return 'Recovery & Sleep'
        elif any(keyword in text_lower for keyword in ['frequency', 'times per week']):
            return 'Training Frequency'
        elif any(keyword in text_lower for keyword in ['exercise', 'selection', 'compound', 'isolation']):
            return 'Exercise Selection'
        elif any(keyword in text_lower for keyword in ['mechanical tension', 'muscle length', 'stretched']):
            return 'Hypertrophy Mechanisms'
        elif any(keyword in text_lower for keyword in ['effective reps', 'motor unit']):
            return 'Effective Reps Concept'
        elif any(keyword in text_lower for keyword in ['periodization', 'autoregulation', 'block']):
            return 'Advanced Programming'
        elif any(keyword in text_lower for keyword in ['regional', 'fiber type']):
            return 'Muscle Physiology'
        else:
            return 'General Hypertrophy'
    
    def get_learning_recommendations(self, wrong_answers: List[Dict]) -> Dict:
        """
        Generate learning recommendations based on wrong answers
        Returns: Dictionary with grouped mistakes and study recommendations
        """
        if not wrong_answers:
            return {'message': 'Perfect score! No areas need improvement.'}
        
        # Group by concept area
        grouped_mistakes = {}
        for answer in wrong_answers:
            concept = answer['concept_area']
            if concept not in grouped_mistakes:
                grouped_mistakes[concept] = []
            grouped_mistakes[concept].append(answer)
        
        # Sort by number of mistakes (prioritize problem areas)
        sorted_concepts = sorted(
            grouped_mistakes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        recommendations = {
            'total_mistakes': len(wrong_answers),
            'concept_areas': [
                {
                    'concept': concept,
                    'mistake_count': len(mistakes),
                    'questions': mistakes
                }
                for concept, mistakes in sorted_concepts
            ],
            'priority_areas': [concept for concept, _ in sorted_concepts[:3]]  # Top 3
        }
        
        return recommendations
    
    def is_assessment_in_progress(self) -> bool:
        """Check if assessment is currently in progress"""
        return self.current_assessment is not None and self.current_question_index < len(self.shuffled_questions)
    
    def reset(self):
        """Reset assessment engine state"""
        self.current_assessment = None
        self.current_tier = None
        self.current_question_index = 0
        self.used_question_ids.clear()
        self.answers = []
        print("🔄 Assessment engine reset")


# Test function
if __name__ == "__main__":
    print("🧪 Testing AssessmentEngine...")
    
    engine = AssessmentEngine()
    
    # Test starting assessment
    info = engine.start_assessment(0)  # Tier 1
    print(f"\n✅ Started: {info['tier_title']}")
    
    # Test getting first question
    question = engine.get_current_question()
    if question:
        print(f"\n📝 Question {question['question_number']}/{question['total_questions']}:")
        print(f"   {question['question']}")
        print(f"   Options: {len(question['options'])}")
    
    print("\n✅ AssessmentEngine test complete!")
