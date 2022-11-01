# @misc{mromero2021t5-base-finetuned-question-generation-ap,
#   title={T5 (base) fine-tuned on SQUAD for QG via AP},
#   author={Romero, Manuel},
#   publisher={Hugging Face},
#   journal={Hugging Face Hub},
#   howpublished={\url{https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap}},
#   year={2021}
# }
import csv
import sys

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
file_cache = dict()

def get_question(answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'],
                            max_length=max_length)

    return tokenizer.decode(output[0])


def generation(training_file, n_questions=1):
    question_count = 0
    output = open('output.txt', 'w')
    output.close()
    with open(training_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        next(tsv_file)

        for line in tsv_file:
            training_context = line[0]
            answer = line[2]

            if training_context not in file_cache:
                training_context_file = open(training_context[1:], "r")
                context = training_context_file.read().replace('\n', '')
                sentences = list(map(str.strip, context.split(".")))
                file_cache[training_context] = sentences
                training_context_file.close()

            sentences = file_cache[training_context]

            for index, sentence in enumerate(sentences):
                word_list = sentence.split()
                if answer in word_list:
                    question = get_question(answer, sentence)
                    output = open('output.txt', 'a')
                    output.writelines(question + "\n")
                    output.close()
                    question_count += 1
                    break

            if question_count == n_questions:
                break

    file.close()


if __name__ == '__main__':
    training_file = sys.argv[1]
    n_questions = int(sys.argv[2])
    generation(training_file, n_questions)



# context = "Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
# answer = "Manuel"
#
# print(get_question(answer, context))
