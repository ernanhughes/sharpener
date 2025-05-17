# test_integration.py
from sharpener.sharpening_agent import SharpeningAgent

agent = SharpeningAgent(model_name="llama3", evaluator_device='cpu')

test_prompt = "Explain the benefits and limitations of reinforcement learning in AI research."
result = agent.execute({'prompt': test_prompt})

print("Final Sharpened Output:", result['sharpened_output'])
print("Evaluation Scores:", result['scores'])
