

from promptsearch import PromptSearch
from promptsearch import PromptModel
from weave import Object
from weave import Dataset
import traceback
import random
import asyncio
import re
import weave
import json
import openai


# import from parent directory
# run gpt-4 on the hellaswag dataset

weave.init('hellaswag')

dataset_name = "hellaswag-small"
prompt_dataset_name = 'prompt_score_4'
initial_prompt_template = """Here is a story and four possible endings - which do you think is most likely? You should end
    your response with the exact string "the answer: " followed by A, B, C or D.
    Story: {ctx}
    Ending A: {ending0}
    Ending B: {ending1}
    Ending C: {ending2}
    Ending D: {ending3}
"""


@weave.op()
def get_openai_response(prompt: str):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content


def publish_dataset():
    # Load the HellaSWAG dataset
    with open('examples/hellaswag/hellaswag_train.jsonl', 'r') as file:
        # load from a jsonl file
        hellaswag_data = [json.loads(line) for line in file]

    list_of_data_dict = []
    for data in hellaswag_data:
        ctx = data['ctx']
        ending0 = data['endings'][0]
        ending1 = data['endings'][1]
        ending2 = data['endings'][2]
        ending3 = data['endings'][3]
        label = data['label']
        row = {'ctx': ctx, 'ending0': ending0, 'ending1': ending1,
               'ending2': ending2, 'ending3': ending3, 'label': label}

        list_of_data_dict.append(
            row)
    dataset = Dataset(name='hellaswag-small',
                      rows=list_of_data_dict[:10])
    weave.publish(dataset)
    dataset = Dataset(name='hellaswag-100',
                      rows=list_of_data_dict[:100])
    weave.publish(dataset)
    dataset = Dataset(name='hellaswag-train',
                      rows=list_of_data_dict)
    weave.publish(dataset)


class HSModel(PromptModel):
    model_name: str
    prompt_template: str

    @weave.op()
    async def predict(self, ctx: str, ending0: str, ending1: str, ending2: str, ending3: str) -> dict:

        client = openai.AsyncClient()

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.prompt_template.format(
                    ctx=ctx, ending0=ending0, ending1=ending1, ending2=ending2, ending3=ending3)}
            ],
        )
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("No response from model")

        match = re.search(r"the answer: ([ABCD])", result)
        if match:
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            answer = answer_map[match.group(1)]
        else:
            answer = None  # Handle case where no valid answer is found
        parsed = (answer != None)

        return {'pred': answer}


@weave.op()
def score(label: dict, model_output: dict) -> dict:
    return {'correct': label == model_output['pred']}


def eval_prompt(result: dict):
    return result['score']['correct']['true_fraction']


publish_dataset()
dataset = weave.ref(dataset_name).get()

model = HSModel(model_name='gpt-4o', prompt_template=initial_prompt_template)

evaluation = weave.Evaluation(
    dataset=dataset, scorers=[score])

ps = PromptSearch(model=model, dataset=dataset, evaluation=evaluation)
ps.steps(100)
