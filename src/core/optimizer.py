import re
import os
import random
import logging
from src.model.base_client import BaseModelClient

logger = logging.getLogger("OPRO")

class Optimizer:
    def __init__(self, model_client, config):
        self.client = model_client
        self.config = config

    def _load_prompt_template(self) -> str:
        path = getattr(self.config, 'meta_prompt_path', 'prompt/meta_prompt.txt')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load meta prompt from {path}: {e}")
            raise e

    def _bucketize_score(self, score: float, num_buckets: int = 100) -> int:
        return round(score * num_buckets)

    def _format_history_string(self, history: list) -> str:
        """
        Format history according to the paper's 'Text 4' style:
        Precision: {score} <Start>{instruction}</Start>
        """
        score_threshold = getattr(self.config, 'old_instruction_score_threshold', 0.1)
        valid_history = [h for h in history if h['score'] >= score_threshold]
        
        # Sort by score (low to high) to show optimization trajectory
        sorted_history = sorted(valid_history, key=lambda x: x['score'])
        
        max_num = getattr(self.config, 'max_num_instructions_in_prompt', 20)
        selected_history = sorted_history[-max_num:]
        
        history_str = ""
        for item in selected_history:
            score_val = self._bucketize_score(item['score'])
            inst_text = item['instruction']
            # Paper specific format
            history_str += f"Precision: {score_val} <Start>{inst_text}</Start>\n"
            
        return history_str.strip()

    def _format_few_shot_examples(self, dataset: list, wrong_questions_counter: dict = None) -> str:
        """
        Format examples according to the paper:
        Q: {question}
        A: <INS>
        Ground truth answer:
        {answer}
        """
        num_shots = getattr(self.config, 'num_few_shot_questions', 3)
        
        # Simple random selection for exemplars
        selected_data = random.sample(dataset, min(len(dataset), num_shots))
        
        ex_str = ""
        for d in selected_data:
            # Paper specific format with <INS> placeholder
            ex_str += f"Q: {d['input']}\nA: <INS>\nGround truth answer:\n{d['target']}\n\n"
        return ex_str.strip()

    def generate_new_instructions(self, history: list, dataset: list = None, wrong_questions_counter: dict = None) -> list:
        history_str = self._format_history_string(history)
        examples_str = self._format_few_shot_examples(dataset, wrong_questions_counter)
        
        template = self._load_prompt_template()
        meta_prompt = template.format(
            history=history_str,
            few_shot_examples=examples_str
        )
        
        logger.info(f"Meta Prompt:\n{meta_prompt}")

        num_prompts = getattr(self.config, 'num_prompts_to_generate', 8)
        new_instructions = []
        
        for _ in range(num_prompts):
            # Paper suggests high temperature for exploration
            raw_output = self.client.generate_text(meta_prompt)
            parsed = self._extract_instruction(raw_output)
            if parsed:
                new_instructions.append(parsed)
                
        return new_instructions

    def _extract_instruction(self, text: str) -> str:
        """
        Extract content inside <Start>...</Start>
        The model might generate "The answer is...</Start>" since prompt ends with "Output: <Start>"
        """
        # Case 1: Model outputs full tags <Start>...</Start>
        match = re.search(r"<Start>(.*?)</Start>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        # Case 2: Model continues from the prompt's <Start>
        # If the output doesn't start with <Start>, we assume it *is* the content
        # checking for closing tag
        if "</Start>" in text:
            return text.split("</Start>")[0].strip()
            
        return text.strip().replace('<Start>', '').replace('</Start>', '')