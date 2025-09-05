def apply_for_mmlu_redux(sample: dict) -> list:
    hint = "Please answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP)"
    question = sample['question']
    choices = sample['choices']
    choices_template = list()
    for i, c in enumerate(choices):
        choices_template.append(f"{chr(ord('A')+i)}. {c}")
    choices_template = "\n".join(choices_template)
    msg = f"{hint}\nQuestion: {question}\nOptions:\n{choices_template}\n"
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": msg
        }
    ]


def apply_for_mmlu_redux_2(sample: dict) -> list:
    hint = f"There is a single choice question about {sample['config_name']}. Answer the question by replying A, B, C or D in the last line of your response following format: 'ANSWER: $LETTER'"
    question = sample['question']
    choices = sample['choices']
    choices_template = list()
    for i, c in enumerate(choices):
        choices_template.append(f"{chr(ord('A')+i)}. {c}")
    choices_template = "\n".join(choices_template)
    msg = f"{hint}\nQuestion: {question}\nOptions:\n{choices_template}\n"
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": msg
        }
    ]

def apply_for_math500(sample: dict) -> list:
    """Chat template for Math-500 evaluation with Qwen3"""
    problem = sample.get('problem', sample.get('question', ''))
    msg = f"{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
    return [
        {
            "role": "user",
            "content": msg
        }
    ]

def apply_for_livecode_v5(sample: dict) -> list:
    """Chat template for LiveCodeBench V5 evaluation with Qwen3"""
    title = sample.get('question_title', '')
    content = sample.get('question_content', '')
    starter_code = sample.get('starter_code', '')
    
    # Combine title and content for the problem description
    problem = f"{title}\n\n{content}"
    if starter_code:
        problem += f"\n\nStarter Code:\n```python\n{starter_code}\n```"
    
    msg = f"{problem}\n\nProvide only the complete Python code solution."
    return [
        {
            "role": "user",
            "content": msg
        }
    ]


def apply_for_livecodebench(sample: dict) -> list:
    """Chat tempalte for livecodebench following lighteval"""
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nQuestion: {sample['question_content']}\n\n"
    if starter_code := sample.get('starter_code', None):
        prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return [
        {
            "role": "system",
            "content" : "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def apply_for_ifeval(sample: dict) -> list:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": sample['prompt']
        }
    ]


def apply_for_ruler(sample: dict) -> list:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": sample['input']
        }
    ]

def apply_for_aime25(sample: dict) -> list:
    """Chat template for AIME 2025 evaluation"""
    problem = sample['question']
    
    msg = f"This is an AIME problem. The answer must be an integer from 0 to 999.\n\nProblem: {problem}\n\nSolve step by step and put your final answer in \\boxed{{}}."
    
    return [
        {
            "role": "system",
            "content": "You are an expert mathematician."
        },
        {
            "role": "user",
            "content": msg
        }
    ]


def trim_reasoning_tokens(response: str) -> str:
    """Trim reasoning tokens from the response <think>...</think>"""
    res = response.strip()
    if "<think>" in res and "</think>" in res:
        start_idx = res.index("<think>")
        end_idx = res.index("</think>")
        if start_idx < end_idx:
            res = res[end_idx+len("</think>"):]
    return res
