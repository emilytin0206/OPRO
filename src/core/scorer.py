import logging
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Scorer:
    def __init__(self, model_client, config):
        self.client = model_client
        self.config = config

    def score_instruction(self, instruction: str, dataset: list) -> float:
        """
        Score a single instruction (Answer Starting Sentence) on the dataset.
        """
        correct_count = 0
        total = len(dataset)
        
        for item in dataset:
            input_text = item['input']
            target_text = item['target']
            
            # --- PAPER SPECIFIC LOGIC ---
            # The instruction is an "Answer Starting Sentence".
            # It should be appended to the prompt, not prepended.
            # Example Prompt construction:
            # Q: {input}
            # A: {instruction}
            
            # Note: We assume the input_text usually contains "Q: ..." or is just the question.
            # We construct a standard QA format.
            
            full_prompt = f"Q: {input_text}\nA: {instruction} "
            
            # Generate prediction
            prediction = self.client.generate_text(full_prompt).strip()
            
            # --- Evaluation ---
            # Since we provided the start of the answer, the model completes it.
            # For strict exact match scoring on GSM8K, we usually look for the final number.
            # But the 'instruction' helps the reasoning path.
            
            if self._evaluate_prediction(prediction, target_text):
                correct_count += 1
                
        score = correct_count / total if total > 0 else 0.0
        return score

    def _evaluate_prediction(self, prediction: str, target: str) -> bool:
        """
        Simple exact match or containment check.
        For GSM8K, usually we extract the number after '####'.
        Here we implement a robust check.
        """
        # Normalize
        pred_clean = prediction.lower().strip()
        targ_clean = target.lower().strip()
        
        # 1. Direct match
        if targ_clean in pred_clean:
            return True
            
        # 2. Number extraction (GSM8K specific)
        # Assuming target is the final number
        try:
            target_num = self._extract_number(targ_clean)
            pred_nums = self._extract_all_numbers(pred_clean)
            if target_num is not None and target_num in pred_nums:
                return True
        except:
            pass
            
        return False

    def _extract_number(self, text):
        import re
        # Extract the last number in the text, handling commas and decimals
        matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        if matches:
            return float(matches[-1].replace(',', ''))
        return None

    def _extract_all_numbers(self, text):
        import re
        matches = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        return [float(m.replace(',', '')) for m in matches]