import openai
import weave

weave.init('numbers')

# read in train.csv
with open('train.csv', 'r') as f:
    data = f.readlines()[1:]  # Skip the first line (header)
    # extract the questions, numbers and answers column
    questions = [row.split(',')[0] for row in data]
    numbers = [row.split(',')[1] for row in data]
    answers = [row.split(',')[3] for row in data]
    # convert answer strings to numbers
    answers = [float(answer) for answer in answers]

    # split numbers space separated into array
    numbers = [num.split(' ') for num in numbers]

    # insert numbers in the questions - ie number0 becomes first number, number1 becomes second number, etc
    for i in range(len(questions)):
        for j, num in enumerate(numbers[i]):
            questions[i] = questions[i].replace(f'number{j}', num)


@weave.op()
def get_openai_response(question):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}])

    return response.choices[0].message.content


generated_answers = [get_openai_response(
    question) for question in questions[:100]]

for i in range(len(generated_answers)):
    answer = answers[i]
    generated_answer = generated_answers[i]
    print("Answer: " + str(answer))
    print("Generated Answer: " + generated_answer)
    # check if the number in answers shows up in the string generated_answer
    if str(int(answer)) in generated_answer:
        print("True")
    else:
        print("False")
