import weave
from weave import Scorer, WeaveList
from openai import OpenAI

# call evaluation = Evaluation(dataset=dataset, scorers=[LearningScorer(prompt=prompt)])


@weave.op()
def call_model(prompt: str) -> str:

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    model_output = response.choices[0].message.content
    return model_output


class LearningScorer(Scorer):

    prompt: str

    def score(self, model_output: str, labeled_actuals: str) -> dict:
        print(model_output)
        print(labeled_actuals)
        feedback_prompt = """I tried using a prompt on an input and got an output. How could I have improved the system prompt to get a better answer.
                    # Model Output:
                    ####{model_output}####

                    # Correct Answers:
                    ####{labeled_actuals}####

                    # Prompt
                    ####{prompt}####"""
        feedback = call_model(feedback_prompt.format(
            model_output=model_output, labeled_actuals=labeled_actuals, system_prompt=self.system_prompt))
        return {'model_output': model_output, 'labeled_actuals': labeled_actuals, 'feedback': feedback}

    def summarize(self, score_rows: WeaveList):
        all_feedback: str = ''
        for row in score_rows:
            all_feedback += row['feedback']
            all_feedback += '\n'

        summarize_prompt = '''Please summarize these learnings succinctly in 2-4 bullet points
        ###{all_feedback}###'''

        summary_of_all_feedback = call_model(
            summarize_prompt.format(mega_feedback=all_feedback))
        return summary_of_all_feedback
