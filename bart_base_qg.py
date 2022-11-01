import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("voidful/context-only-question-generator")

model = AutoModelForSeq2SeqLM.from_pretrained("voidful/context-only-question-generator")


def get_question(context, max_length=64):
    input_text = "context: %s </s>" % context
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'],
                            max_length=max_length)

    return tokenizer.decode(output[0])


def generation(context, n_questions=1):
    question_count = 0
    output = open('output.txt', 'w')
    output.close()
    text_file = open(context, 'r')
    context_str = text_file.read().replace('\n', '')
    sentences = list(map(str.strip, context_str.split(".")))
    text_file.close()

    while question_count < n_questions:
        if question_count >= len(sentences):
            break
        question = get_question(sentences[question_count])
        output = open('output.txt', 'a')
        output.writelines(question + "\n")
        output.close()
        question_count += 1


if __name__ == '__main__':
    context = sys.argv[1]
    n_questions = int(sys.argv[2])
    generation(context, n_questions)