def apply_for_math500(sample: dict) -> list:
    """Chat template for Math-500 evaluation with GPT-OSS"""
    problem = sample.get('problem', sample.get('question', ''))
    msg = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
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
            "role": "user",
            "content": prompt
        }
    ]


def apply_for_ifeval(sample: dict) -> list:
    return [
        {
            "role": "user",
            "content": sample['prompt']
        }
    ]


def apply_for_mmlu_redux(sample: dict) -> list:
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
            "role": "user",
            "content": msg
        }
    ]


def apply_for_ruler(sample: dict) -> list:
    return [
        {
            "role": "user",
            "content": sample['input']
        }
    ]
