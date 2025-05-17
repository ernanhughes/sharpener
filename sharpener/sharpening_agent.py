# sharpening_agent.py
from co_ai.base_agent import BaseAgent
from mrq_text_evaluator import MRQSelfEvaluator
import dspy

class SharpeningAgent(BaseAgent):
    def __init__(self, model_name="llama3", evaluator_device='cpu'):
        super().__init__()
        self.llm = dspy.OllamaLocal(model=model_name)
        self.evaluator = MRQSelfEvaluator(device=evaluator_device)

    def generate_output(self, prompt):
        response = self.llm(prompt).completion
        return response

    def sharpen(self, prompt, iterations=3):
        current_output = self.generate_output(prompt)
        
        for i in range(iterations):
            sharpen_prompt = f"""You are a critical AI editor. Please improve the clarity, detail, and accuracy of the following response:

Prompt: {prompt}

Current Response: {current_output}

Improved Response:"""

            improved_output = self.generate_output(sharpen_prompt)

            preferred_output, scores = self.evaluator.evaluate(
                prompt, current_output, improved_output
            )

            print(f"Iteration {i+1} | Current Score: {scores['value_a']:.3f}, Improved Score: {scores['value_b']:.3f}")

            if preferred_output == current_output:
                print("No further improvement detected. Stopping sharpening loop.")
                break

            current_output = improved_output

        return current_output, scores

    def execute(self, inputs):
        prompt = inputs['prompt']
        sharpened_output, scores = self.sharpen(prompt, iterations=3)

        log_entry = {
            "prompt": prompt,
            "sharpened_output": sharpened_output,
            "scores": scores
        }

        self.logger.log("SharpeningWithMRQEvaluation", log_entry)

        return log_entry
