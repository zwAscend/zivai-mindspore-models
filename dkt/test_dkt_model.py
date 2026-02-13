# File: test_dkt_model.py
"""
Comprehensive DKT Model Testing and Evaluation
"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import pickle
import json
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def resolve_work_dir() -> Path:
    """Resolve the local dkt directory that contains model/data artifacts."""
    candidates = []
    try:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
    except NameError:
        # __file__ is not defined in notebooks/ipykernel.
        script_dir = None

    candidates.extend([Path.cwd() / "dkt", Path.cwd()])

    for candidate in candidates:
        if (candidate / "assistments_2009_2010.csv").exists():
            return candidate
    return script_dir if script_dir is not None else Path.cwd()


WORK_DIR = resolve_work_dir()

class DKTModelTester:
    """
    Comprehensive tester for trained DKT model
    """
    
    def __init__(self, model_path: str = "dkt_model_full.mindir",
                 skill_mapping_path: str = "skill_mapping_full.pkl",
                 dataset_path: str = "assistments_2009_2010.csv"):
        """
        Initialize the tester with trained model
        """
        print("="*70)
        print("DKT MODEL TESTING & EVALUATION SUITE")
        print("="*70)
        
        # Resolve local paths relative to dkt directory
        self.work_dir = WORK_DIR
        self.dataset_path = self.work_dir / dataset_path
        self.skill_mapping_path = self.work_dir / skill_mapping_path

        # Pick first available model path
        model_candidates = [self.work_dir / model_path, self.work_dir / "dkt_model_full.mindir", self.work_dir / "dkt_model.mindir"]
        self.model_path = next((p for p in model_candidates if p.exists()), model_candidates[0])
        self.model_output_dim = None

        print(f"   Working directory: {self.work_dir}")
        print(f"   Dataset path: {self.dataset_path}")
        if not self.dataset_path.exists():
            print("   ⚠ Dataset file not found in working directory")

        # Load model
        print("\n1. Loading trained model...")
        loaded_graph = ms.load(str(self.model_path))
        # MindIR load returns a FuncGraph in many MindSpore versions; wrap for inference.
        self.model = nn.GraphCell(loaded_graph)
        self.model.set_train(False)
        print(f"   ✓ Model loaded from {self.model_path}")
        
        # Load skill mapping
        print("\n2. Loading skill mapping...")
        with open(self.skill_mapping_path, 'rb') as f:
            self.skill_to_idx = pickle.load(f)
        
        # Create reverse mapping
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}
        self.num_skills = len(self.skill_to_idx)
        print(f"   ✓ Loaded {self.num_skills:,} skills")
        
        # Student history storage for testing
        self.test_students = {}
        
        print("\n" + "="*70)
        print("TESTER READY!")
        print("="*70)

    def _switch_to_fallback_model(self) -> bool:
        """Try a smaller local MindIR if the primary model OOMs on GPU."""
        fallback_path = self.work_dir / "dkt_model.mindir"
        if fallback_path.exists() and fallback_path != self.model_path:
            print(f"Switching evaluation model to fallback: {fallback_path}")
            loaded_graph = ms.load(str(fallback_path))
            self.model = nn.GraphCell(loaded_graph)
            self.model.set_train(False)
            self.model_path = fallback_path
            self.model_output_dim = None
            return True
        return False
    
    # ==================== PART 1: STUDENT INTERACTION TESTS ====================
    
    def simulate_student_interactions(self):
        """
        Simulate realistic student learning journeys
        """
        print("\n" + "="*70)
        print("PART 1: STUDENT INTERACTION SIMULATIONS")
        print("="*70)
        
        # Test Case 1: Student learning linear equations
        print("\n📚 TEST CASE 1: Student Learning Linear Equations")
        print("-" * 50)
        
        student_1 = StudentSimulator("student_001", self)
        
        # Simulate 5 interactions with linear equations
        interactions_1 = [
            {"skill": "linear_equations", "correct": 0, "description": "First attempt - incorrect"},
            {"skill": "linear_equations", "correct": 0, "description": "Second attempt - still incorrect"},
            {"skill": "linear_equations", "correct": 1, "description": "Third attempt - correct!"},
            {"skill": "linear_equations", "correct": 1, "description": "Practice - correct"},
            {"skill": "linear_equations", "correct": 1, "description": "Mastered - correct"}
        ]
        
        for i, interaction in enumerate(interactions_1, 1):
            student_1.add_interaction(
                skill_id=interaction["skill"],
                correct=interaction["correct"],
                description=interaction["description"]
            )
        
        # Get mastery after each interaction
        student_1.show_mastery_progression()
        
        # Test Case 2: Student learning multiple topics
        print("\n📚 TEST CASE 2: Student Learning Multiple Topics")
        print("-" * 50)
        
        student_2 = StudentSimulator("student_002", self)
        
        # Simulate mixed topic learning
        interactions_2 = [
            {"skill": "linear_equations", "correct": 1, "topic": "Algebra"},
            {"skill": "quadratics", "correct": 0, "topic": "Algebra"},
            {"skill": "linear_equations", "correct": 1, "topic": "Algebra"},
            {"skill": "pythagoras", "correct": 1, "topic": "Geometry"},
            {"skill": "quadratics", "correct": 1, "topic": "Algebra"},
            {"skill": "circle_theorems", "correct": 0, "topic": "Geometry"},
            {"skill": "pythagoras", "correct": 1, "topic": "Geometry"},
            {"skill": "quadratics", "correct": 1, "topic": "Algebra"},
        ]
        
        for interaction in interactions_2:
            student_2.add_interaction(
                skill_id=interaction["skill"],
                correct=interaction["correct"],
                description=f"{interaction['topic']} - {'Correct' if interaction['correct'] else 'Incorrect'}"
            )
        
        # Show final mastery
        student_2.show_current_mastery()
        
        # Test Case 3: Student struggling with a concept
        print("\n📚 TEST CASE 3: Student Struggling with Quadratics")
        print("-" * 50)
        
        student_3 = StudentSimulator("student_003", self)
        
        # Simulate struggle pattern
        interactions_3 = [
            {"skill": "quadratics", "correct": 0, "attempt": 1},
            {"skill": "quadratics", "correct": 0, "attempt": 2},
            {"skill": "quadratics", "correct": 0, "attempt": 3},
            {"skill": "linear_equations", "correct": 1, "attempt": 1},  # Good at this
            {"skill": "quadratics", "correct": 1, "attempt": 4},  # Finally gets it
            {"skill": "quadratics", "correct": 1, "attempt": 5},  # Reinforces
        ]
        
        for interaction in interactions_3:
            student_3.add_interaction(
                skill_id=interaction["skill"],
                correct=interaction["correct"],
                description=f"Attempt {interaction['attempt']}"
            )
        
        # Show how mastery changed
        student_3.show_struggle_recovery()
        
        return {
            "student_1": student_1,
            "student_2": student_2,
            "student_3": student_3
        }
    
    # ==================== PART 2: SKILL ANALYSIS ====================
    
    def analyze_skill_relationships(self):
        """
        Analyze how skills relate to each other in the model
        """
        print("\n" + "="*70)
        print("PART 2: SKILL RELATIONSHIP ANALYSIS")
        print("="*70)
        
        # Get sample skills for analysis
        sample_skills = list(self.skill_to_idx.keys())[:10]
        
        print(f"\nAnalyzing relationships between {len(sample_skills)} sample skills:")
        for skill in sample_skills:
            print(f"  - {skill}")
        
        # Create a student who has mastered some skills
        student = StudentSimulator("analysis_student", self)
        
        # Master algebra skills
        for skill in ["linear_equations", "quadratics", "inequalities"]:
            if skill in self.skill_to_idx:
                for _ in range(3):  # 3 correct attempts
                    student.add_interaction(skill, 1, "practice")
        
        # Get predictions for related skills
        mastery = student.get_current_mastery()
        
        print("\nSkill Mastery Heat Map:")
        print("-" * 50)
        print(f"{'Skill':<25} {'Mastery':<10} {'Status':<15}")
        print("-" * 50)
        
        for skill, prob in sorted(mastery.items(), key=lambda x: x[1], reverse=True)[:15]:
            status = "MASTERED" if prob > 0.7 else "LEARNING" if prob > 0.4 else "WEAK"
            print(f"{skill:<25} {prob:.3f}     {status:<15}")
    
    # ==================== PART 3: EVALUATION METRICS ====================
    
    def evaluate_model_performance(self, test_sequences_path: str = "test_sequences_full.pkl"):
        """
        Comprehensive evaluation with multiple metrics
        """
        print("\n" + "="*70)
        print("PART 3: COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Load test sequences
        print("\nLoading test sequences...")
        resolved_test_sequences = self.work_dir / test_sequences_path
        with open(resolved_test_sequences, 'rb') as f:
            test_sequences = pickle.load(f)
        
        print(f"Loaded {len(test_sequences)} test sequences")
        
        # Build evaluation windows first, then run batched inference to reduce GPU pressure
        input_windows = []
        target_windows = []
        max_sequences = 100
        max_windows = 6000
        max_len = 99
        
        print("\nPreparing evaluation windows...")
        for seq in test_sequences[:max_sequences]:
            if len(seq) < 5:
                continue
            for i in range(1, len(seq) - 1):
                history = seq[:i]
                next_token = seq[i]
                target_skill = (next_token - 1) // 2
                if target_skill >= self.num_skills:
                    continue
                input_seq = np.zeros(max_len, dtype=np.int32)
                input_seq[-min(len(history), max_len):] = history[-max_len:]
                input_windows.append(input_seq)
                target_windows.append(target_skill)
                if len(input_windows) >= max_windows:
                    break
            if len(input_windows) >= max_windows:
                break
        
        if not input_windows:
            print("No valid evaluation windows generated.")
            return {}
        
        print(f"Evaluating on {len(input_windows)} windows...")
        print("Using batch size 1 for MindIR shape compatibility ([1, 99])")
        
        def run_single_batch_inference():
            preds = []
            tgts = []
            for i, (window, tgt) in enumerate(zip(input_windows, target_windows), 1):
                # MindIR was exported with fixed input shape [1, 99]
                if self.model_output_dim is not None:
                    window = np.clip(window, 0, self.model_output_dim * 2)
                input_tensor = ms.Tensor(window.reshape(1, -1), ms.int32)
                pred = self.model(input_tensor).asnumpy()
                if self.model_output_dim is None:
                    self.model_output_dim = int(pred.shape[1])
                preds.append(pred)
                tgts.append(tgt)
                if i % 1000 == 0:
                    print(f"  Processed {i}/{len(input_windows)} windows")
            return np.concatenate(preds, axis=0), np.array(tgts, dtype=np.int32)
        
        try:
            predictions, targets = run_single_batch_inference()
        except RuntimeError as e:
            err_msg = str(e)
            if "CUDNN_STATUS_ALLOC_FAILED" in err_msg or "cuDNN Error" in err_msg:
                print("\nGPU memory allocation failed during evaluation.")
                switched = self._switch_to_fallback_model()
                if switched:
                    predictions, targets = run_single_batch_inference()
                    print("Evaluation completed with fallback model.")
                else:
                    raise RuntimeError(
                        "GPU OOM during evaluation and no fallback model available. "
                        "Restart the kernel, close other GPU workloads, and retry."
                    ) from e
            else:
                raise
        
        # Guard against mismatched mapping/model output dimensions
        output_dim = predictions.shape[1]
        valid_mask = targets < output_dim
        dropped = int(np.sum(~valid_mask))
        if dropped > 0:
            print(f"Dropping {dropped} samples with target index >= model output dim ({output_dim})")
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]
        
        if len(predictions) == 0:
            print("No valid predictions after filtering.")
            return {}
        
        print(f"\nEvaluated on {len(predictions)} predictions")
        
        # ===== METRIC 1: AUC (Area Under ROC Curve) =====
        print("\n" + "="*50)
        print("METRIC 1: AUC (Area Under ROC Curve)")
        print("="*50)
        
        from sklearn.metrics import roc_auc_score
        
        # For each skill, calculate AUC
        skill_aucs = []
        for skill_idx in range(min(50, self.num_skills)):  # Top 50 skills
            # Create binary labels for this skill
            y_true = (targets == skill_idx).astype(int)
            y_score = predictions[:, skill_idx]
            
            if len(np.unique(y_true)) > 1:  # Need both classes
                try:
                    auc = roc_auc_score(y_true, y_score)
                    skill_aucs.append(auc)
                except:
                    pass
        
        if skill_aucs:
            avg_auc = np.mean(skill_aucs)
            print(f"Average AUC across skills: {avg_auc:.4f}")
            print(f"Range: {np.min(skill_aucs):.4f} - {np.max(skill_aucs):.4f}")
        else:
            avg_auc = float("nan")
            print("Average AUC across skills: N/A (insufficient class variety in sampled windows)")
        print("\nInterpretation:")
        print("  > 0.9: Excellent discrimination")
        print("  > 0.8: Good discrimination")
        print("  > 0.7: Fair discrimination")
        print(f"  Your model: {'Excellent' if avg_auc>0.9 else 'Good' if avg_auc>0.8 else 'Fair' if avg_auc>0.7 else 'Needs improvement'}")
        
        # ===== METRIC 2: RMSE (Root Mean Square Error) =====
        print("\n" + "="*50)
        print("METRIC 2: RMSE (Root Mean Square Error)")
        print("="*50)
        
        from sklearn.metrics import mean_squared_error
        
        # Get predictions for actual next skills
        skill_probs = []
        for i in range(len(predictions)):
            skill_probs.append(predictions[i, targets[i]])
        
        skill_probs = np.array(skill_probs)
        true_correct = np.ones_like(skill_probs)  # We're predicting next skill
        
        rmse = np.sqrt(mean_squared_error(true_correct, skill_probs))
        print(f"RMSE: {rmse:.4f}")
        print("\nInterpretation:")
        print("  0.0: Perfect predictions")
        print("  <0.3: Very good")
        print("  0.3-0.5: Acceptable")
        print(f"  Your model: {'Very good' if rmse<0.3 else 'Acceptable' if rmse<0.5 else 'High error'}")
        
        # ===== METRIC 3: PRECISION@K =====
        print("\n" + "="*50)
        print("METRIC 3: PRECISION@K")
        print("="*50)
        print("How often the correct skill is in top K predictions")
        
        precision_at_k = {}
        for k in [1, 3, 5, 10]:
            correct = 0
            for i in range(len(predictions)):
                top_k = np.argsort(predictions[i])[-k:][::-1]
                if targets[i] in top_k:
                    correct += 1
            
            precision = correct / len(predictions)
            precision_at_k[k] = precision
            print(f"Precision@{k}: {precision:.4f}")
        
        print("\nInterpretation:")
        print("  High precision@1 means model knows exactly the next skill")
        print("  High precision@5 means model identifies the right skill area")
        
        # ===== METRIC 4: CALIBRATION ERROR =====
        print("\n" + "="*50)
        print("METRIC 4: CALIBRATION ERROR")
        print("="*50)
        print("How well do predicted probabilities match actual correctness?")
        
        # Group predictions by confidence bins
        bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(bins)-1):
            bin_mask = (skill_probs >= bins[i]) & (skill_probs < bins[i+1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(true_correct[bin_mask])
                bin_confidence = np.mean(skill_probs[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                
                print(f"Bin {bins[i]:.1f}-{bins[i+1]:.1f}: "
                      f"Confidence={bin_confidence:.3f}, "
                      f"Accuracy={bin_accuracy:.3f}")
        
        if bin_accuracies and bin_confidences:
            calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
            print(f"\nCalibration Error: {calibration_error:.4f}")
            print("Interpretation:")
            print("  <0.05: Well calibrated")
            print("  0.05-0.10: Moderately calibrated")
            print("  >0.10: Poorly calibrated")
        
        # ===== METRIC 5: SKILL SEPARATION =====
        print("\n" + "="*50)
        print("METRIC 5: SKILL SEPARATION")
        print("="*50)
        print("Can the model distinguish between different skills?")
        
        # Pick two random skills
        skill1, skill2 = np.random.choice(self.num_skills, 2, replace=False)
        
        # Get predictions where these skills were the target
        preds_skill1 = predictions[targets == skill1][:, skill1]
        preds_skill2 = predictions[targets == skill2][:, skill2]
        
        if len(preds_skill1) > 0 and len(preds_skill2) > 0:
            mean1 = np.mean(preds_skill1)
            mean2 = np.mean(preds_skill2)
            separation = abs(mean1 - mean2)
            
            print(f"Skill {skill1}: Average mastery {mean1:.3f}")
            print(f"Skill {skill2}: Average mastery {mean2:.3f}")
            print(f"Separation: {separation:.3f}")
            print("\nInterpretation:")
            print("  >0.3: Good separation")
            print("  0.2-0.3: Moderate separation")
            print("  <0.2: Poor separation (skills may be similar)")
        
        return {
            "avg_auc": avg_auc,
            "rmse": rmse,
            "precision_at_k": precision_at_k,
            "calibration_error": calibration_error if 'calibration_error' in locals() else None
        }
    
    # ==================== PART 4: RECOMMENDATION TESTS ====================
    
    def test_recommendation_system(self):
        """
        Test how well the model recommends next skills
        """
        print("\n" + "="*70)
        print("PART 4: RECOMMENDATION SYSTEM TEST")
        print("="*70)
        
        student = StudentSimulator("rec_student", self)
        
        # Student shows they're good at basic algebra
        for skill in ["linear_equations", "linear_equations", "linear_equations"]:
            student.add_interaction("linear_equations", 1, "good at basics")
        
        # Get recommendations
        weak_skills = student.get_weak_skills(n=5)
        recommendations = student.get_next_skill_recommendations()
        
        print("\nBased on student's learning history:")
        print("-" * 50)
        print("Current strengths:")
        for skill, prob in student.get_current_mastery().items():
            if prob > 0.7:
                print(f"  ✓ {skill}: {prob:.3f}")
        
        print("\nRecommended next skills to learn:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['skill']} (Readiness: {rec['readiness']:.1%})")
            print(f"     Reason: {rec['reason']}")
        
        print("\nSkills needing review:")
        for skill in weak_skills:
            print(f"  ⚠ {skill['skill_id']}: {skill['mastery']:.1%} mastery")
    
    # ==================== PART 5: VISUALIZATION ====================
    
    def visualize_learning_curves(self, student_id: str = "student_001"):
        """
        Create visualizations of learning progress
        """
        print("\n" + "="*70)
        print("PART 5: LEARNING CURVE VISUALIZATION")
        print("="*70)
        
        # Check if we have the student
        if student_id not in self.test_students:
            print(f"Student {student_id} not found. Run simulation first.")
            return
        
        student = self.test_students[student_id]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Mastery over time
        plt.subplot(2, 2, 1)
        skills_to_plot = list(student.mastery_history.keys())[:5]  # First 5 skills
        for skill in skills_to_plot:
            history = student.mastery_history[skill]
            plt.plot(history, label=skill[:20])
        plt.xlabel('Interaction Number')
        plt.ylabel('Mastery Probability')
        plt.title('Skill Mastery Over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Current mastery distribution
        plt.subplot(2, 2, 2)
        current_mastery = list(student.mastery_history.keys())
        current_values = [student.mastery_history[s][-1] for s in current_mastery[:10]]
        plt.barh(range(len(current_values)), current_values)
        plt.yticks(range(len(current_values)), [s[:15] for s in current_mastery[:10]])
        plt.xlabel('Mastery Probability')
        plt.title('Current Mastery Levels')
        
        # Plot 3: Weak skills radar
        plt.subplot(2, 2, 3)
        weak_skills = student.get_weak_skills(n=5)
        if weak_skills:
            skills = [w['skill_id'][:10] for w in weak_skills]
            mastery = [w['mastery'] for w in weak_skills]
            plt.bar(skills, mastery, color='orange')
            plt.axhline(y=0.6, color='r', linestyle='--', label='Proficiency threshold')
            plt.xlabel('Skills')
            plt.ylabel('Mastery')
            plt.title('Skills Needing Improvement')
            plt.legend()
        
        # Plot 4: Success probability
        plt.subplot(2, 2, 4)
        interactions = student.interaction_history
        if interactions:
            correct_history = [1 if 'correct' in i['description'] else 0 for i in interactions]
            plt.plot(np.cumsum(correct_history) / (np.arange(len(correct_history)) + 1))
            plt.xlabel('Interaction Number')
            plt.ylabel('Cumulative Success Rate')
            plt.title('Learning Progress')
            plt.grid(True)
        
        plt.tight_layout()
        out_path = self.work_dir / "dkt_learning_curves.png"
        plt.savefig(out_path)
        print(f"✓ Visualization saved to '{out_path}'")
        plt.show()


class StudentSimulator:
    """
    Simulate a student for testing
    """
    
    def __init__(self, student_id: str, tester: DKTModelTester):
        self.student_id = student_id
        self.tester = tester
        self.interaction_history = []
        self.mastery_history = defaultdict(list)
        tester.test_students[student_id] = self
    
    def add_interaction(self, skill_id: str, correct: int, description: str = ""):
        """
        Add a student interaction
        """
        # Record interaction
        interaction = {
            "skill": skill_id,
            "correct": correct,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.interaction_history.append(interaction)
        
        # Update DKT (simulate - in reality, this would call your model)
        self._update_model_mastery(skill_id, correct)
    
    def _update_model_mastery(self, skill_id: str, correct: int):
        """
        Get updated mastery from model
        """
        if skill_id not in self.tester.skill_to_idx:
            return
        
        # Convert to token
        skill_idx = self.tester.skill_to_idx[skill_id]
        token = (skill_idx - 1) * 2 + correct + 1
        
        # Prepare input sequence
        history_tokens = []
        for interaction in self.interaction_history:
            s = interaction["skill"]
            c = interaction["correct"]
            if s in self.tester.skill_to_idx:
                s_idx = self.tester.skill_to_idx[s]
                t = (s_idx - 1) * 2 + c + 1
                history_tokens.append(t)
        
        max_len = 99
        input_seq = np.zeros(max_len, dtype=np.int32)
        if history_tokens:
            input_seq[-min(len(history_tokens), max_len):] = history_tokens[-max_len:]
        
        # Get prediction
        input_tensor = ms.Tensor(input_seq.reshape(1, -1), ms.int32)
        mastery = self.tester.model(input_tensor).asnumpy().flatten()
        
        # Update history for all skills
        for idx, prob in enumerate(mastery):
            if idx + 1 in self.tester.idx_to_skill:
                skill_name = self.tester.idx_to_skill[idx + 1]
                self.mastery_history[skill_name].append(prob)
    
    def get_current_mastery(self) -> Dict[str, float]:
        """
        Get current mastery for all skills
        """
        mastery = {}
        for skill, history in self.mastery_history.items():
            if history:
                mastery[skill] = history[-1]
        return mastery
    
    def get_weak_skills(self, n: int = 5) -> List[Dict]:
        """
        Get weakest skills
        """
        mastery = self.get_current_mastery()
        weak = [(skill, prob) for skill, prob in mastery.items() if prob < 0.6]
        weak.sort(key=lambda x: x[1])
        
        return [
            {"skill_id": skill, "mastery": prob}
            for skill, prob in weak[:n]
        ]
    
    def get_next_skill_recommendations(self) -> List[Dict]:
        """
        Get recommendations for next skills to learn
        """
        mastery = self.get_current_mastery()
        
        # Skills with moderate mastery (0.3-0.6) are prime for learning
        recommendations = []
        for skill, prob in mastery.items():
            if 0.3 <= prob <= 0.6:
                recommendations.append({
                    "skill": skill,
                    "readiness": prob,
                    "reason": f"Good foundation ({prob:.0%} mastery), ready to advance"
                })
        
        # Sort by readiness
        recommendations.sort(key=lambda x: x["readiness"], reverse=True)
        return recommendations[:5]
    
    def show_mastery_progression(self):
        """
        Show how mastery changed over time
        """
        print(f"\nStudent {self.student_id} - Learning Progression:")
        print("-" * 50)
        
        for i, interaction in enumerate(self.interaction_history, 1):
            skill = interaction["skill"]
            correct = "✓" if interaction["correct"] else "✗"
            desc = interaction["description"]
            
            # Get mastery after this interaction
            mastery = None
            if skill in self.mastery_history and len(self.mastery_history[skill]) >= i:
                mastery = self.mastery_history[skill][i-1]
            
            mastery_str = f" (mastery: {mastery:.3f})" if mastery else ""
            print(f"  {i:2d}. [{correct}] {skill} - {desc}{mastery_str}")
    
    def show_current_mastery(self):
        """
        Show current mastery levels
        """
        mastery = self.get_current_mastery()
        
        print(f"\nStudent {self.student_id} - Current Mastery:")
        print("-" * 50)
        print(f"{'Skill':<30} {'Mastery':<10} {'Status':<15}")
        print("-" * 50)
        
        sorted_mastery = sorted(mastery.items(), key=lambda x: x[1], reverse=True)
        
        for skill, prob in sorted_mastery[:10]:
            if prob > 0.7:
                status = "✅ MASTERED"
            elif prob > 0.4:
                status = "📚 LEARNING"
            else:
                status = "⚠️ NEEDS WORK"
            
            print(f"{skill:<30} {prob:.3f}     {status:<15}")
    
    def show_struggle_recovery(self):
        """
        Show how student recovered from struggling
        """
        print(f"\nStudent {self.student_id} - Struggle and Recovery:")
        print("-" * 50)
        
        # Track one skill's progression
        skill_to_track = None
        for skill in self.mastery_history.keys():
            if len(self.mastery_history[skill]) > 3:
                skill_to_track = skill
                break
        
        if skill_to_track:
            history = self.mastery_history[skill_to_track]
            print(f"\nSkill: {skill_to_track}")
            print("Interaction | Mastery | Progress")
            print("-" * 40)
            
            for i, mastery in enumerate(history, 1):
                arrow = "↑" if i > 0 and mastery > history[i-1] else "↓" if i > 0 and mastery < history[i-1] else "→"
                print(f"     {i:2d}     |  {mastery:.3f}  |   {arrow}")


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main test execution
    """
    print("\n" + "🎯"*35)
    print("DKT MODEL TESTING AND EVALUATION".center(70))
    print("🎯"*35)
    
    # Force GPU execution
    ms.set_context(device_target="GPU")
    print(f"Using MindSpore device target: {ms.get_context('device_target')}")

    # Initialize tester
    tester = DKTModelTester(
        model_path="dkt_model_full.mindir",
        skill_mapping_path="skill_mapping_full.pkl",
        dataset_path="assistments_2009_2010.csv"
    )
    
    # Run tests
    print("\n" + "📊"*35)
    print("RUNNING COMPREHENSIVE TESTS".center(70))
    print("📊"*35)
    
    # Part 1: Student interactions
    students = tester.simulate_student_interactions()
    
    # Part 2: Skill analysis
    tester.analyze_skill_relationships()
    
    # Part 3: Model evaluation
    if input("\nRun comprehensive evaluation? (y/n): ").lower() == 'y':
        metrics = tester.evaluate_model_performance()
    
    # Part 4: Recommendation test
    tester.test_recommendation_system()
    
    # Part 5: Visualization
    if input("\nGenerate visualizations? (y/n): ").lower() == 'y':
        tester.visualize_learning_curves()
    
    print("\n" + "✨"*35)
    print("TESTING COMPLETE!".center(70))
    print("✨"*35)
    print("\nWhat you've learned about your DKT model:")
    print("  ✓ How it tracks student mastery over time")
    print("  ✓ Which skills it can predict accurately")
    print("  ✓ How to interpret its predictions")
    print("  ✓ How to use it for recommendations")
    print("  ✓ Its strengths and limitations")


if __name__ == "__main__":
    main()
