
# run gpt-4 on the hellaswag dataset
import openai
import json
import weave
import re
import asyncio
from weave import Dataset

weave.init('hellaswag')


@weave.op()
def get_openai_response(prompt: str):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content


def publish_dataset():
    # Load the HellaSWAG dataset
    with open('hellaswag_train.jsonl', 'r') as file:
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
    print(list_of_data_dict)
    dataset = Dataset(name='hellaswag-small', rows=list_of_data_dict[:10])
    weave.publish(dataset)
    dataset = Dataset(name='hellaswag-100', rows=list_of_data_dict[:100])
    weave.publish(dataset)
    dataset = Dataset(name='hellaswag-train', rows=list_of_data_dict)
    weave.publish(dataset)


class HSModel(weave.Model):
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


# @weave.op()
# def run_gpt(ctx, ending0, ending1, ending2, ending3, label):

#     response = get_openai_response(prompt)

#     correct = (answer == label)
#     return parsed, correct, response
@weave.op()
def score(label: dict, model_output: dict) -> dict:
    return {'correct': label == model_output['pred']}


# print(asyncio.run(evaluation.evaluate(model)))

def run_gpt4_on_hellaswag(prompt_template):
    dataset = weave.ref('hellaswag-100').get()

    model = HSModel(model_name='gpt-4o', prompt_template=prompt_template)
    # print(dataset.rows)
    # data = dataset.rows[0]
    print(dataset.rows[0])

    evaluation = weave.Evaluation(
        dataset=dataset, scorers=[score])

    # # Load the HellaSWAG dataset
    # with open('hellaswag_test.jsonl', 'r') as file:
    #     # load from a jsonl file
    #     hellaswag_data = [json.loads(line) for line in file]
    # for data in dataset_ref.rows:
    #     ctx = data['ctx']
    #     ending0 = data['ending0']
    #     ending1 = data['ending1']
    #     ending2 = data['ending2']
    #     ending3 = data['ending3']
    #     label = data['label']

    #     print(asyncio.run(model.predict(ctx, ending0, ending1, ending2, ending3)))

    # Save or process the results as needed
    print(asyncio.run(evaluation.evaluate(model)))


def mutate_template(prompt_template):
    mutate_prompt_template = """Given the following prompt for an LLM, write me a better prompt. I'm going to replace {{ctx}} with the story and {{ending0}} with the first ending etc. Initial prompt = {prompt_template}"""
    new_prompt_template = get_openai_response(
        mutate_prompt_template.format(prompt_template=prompt_template))
    print(new_prompt_template)
    return new_prompt_template
# publish_dataset()


def prompt_search():
    prompt_template = """Here is a story and four possible endings - which do you think is most likely? You should end
        your response with the exact string "the answer: " followed by A, B, C or D.
        Story: {ctx}
        Ending A: {ending0}
        Ending B: {ending1}
        Ending C: {ending2}
        Ending D: {ending3}
        """

    for i in range(10):
        new_prompt_template = mutate_template(prompt_template)
        run_gpt4_on_hellaswag(new_prompt_template)


prompt_search()
