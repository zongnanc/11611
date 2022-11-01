## T5 Base Pretrained Finely Tuned Question Generator

This is a working Question Generator (QG) based on pretrained T5 base finetuned QG
model from HuggingFace (source: mrm8488/t5-base-finetuned-question-generation-ap).
The QG model takes an answer to the question and the context as inputs and will
generate a human readable question. The context has a default constraint of 64 words
in each generated question and is configurable. Currently, the working QG takes in
each single sentence as the context for each question. But later on, the QG system will
be expanded to accept preceeding and following sentences in order to generate question
with level of difficulty. Since answer is a required input to each question generation,
boolean answer such as yes/no will not be able to generate its corresponding question.
The generated questions can be found at output.txt.

**Command to run the code:**

python t5_base_qg.py context_file_name.tsv n_questions

**Note:**

Prepare context directories at the same file path

## Bart Base Question Generator

This is a working Question Generator (QG) based on pretrained Bart-Base sequence to
sequence model from HuggingFace (source:voidful/context-only-question-generator). The
QG model takes in only context as input and generate a human readable question. Currently,
the working QG takes in each single sentence as the context for each question. But later 
on, the QG system will be expanded to accept preceeding and following sentences in order 
to generate question with level of difficulty. The generated questions can be found at 
output.txt.

**Command to run the code:**

python bart_base_qg.py context_file_name.txt n_questions

**Note:**

Prepare context directories at the same file path