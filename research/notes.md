# Trained model notes
The best models so far are emboldend
## AE
* 7b-chatAE00: Specific prompt. No custom tokens. Produces relatively good extracted answers, but sometimes duplicated ones. Does not stop short.
* 7b-chatAE01: Specific prompt. No custom tokens. Further de-duplicated data. Produces decent answers w/ less duplication. Does not stop short.
* 7b-chatAE02: Specific prompt. Custom tokens. Produces trash. Does not stop short.
* **7b-chatAE03**: Less specific prompt. No custom tokens. Produces good answers. Does not stop short.

## QG
* 7b-chatQG00: Relatively specific unoptimized prompt. No custom tokens. Produces decent questions. Spits out a ton of "do not confuse w/ vs #"
* 7b-chatQG01: Relatively specific unoptimized prompt. No custom tokens. Removed "do not confuse" from training data. Produces good questions, though they are a bit short. Spits out trash after the question.
* 7b-chatQG02: Relatively specific unoptimized prompt. Custom tokens. Produces decent questions, though those that are longer are too long. Spits out trash after the question.
* **7b-chatQG03**: Unspecific prompt. No custom tokens. Produces good questions 80% of the time but then continues with trash.







# Input format testing

## Text-completion models 
### AE
<s> ### Here is a context sentence. ### Verse: <context> ### Here are seven nouns and noun phrases in a comma-separated list that appear in this Bible verse:
## Text-completion models QAG


## Chat Models
Examples of how to format:
https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
https://github.com/facebookresearch/llama/issues/481

<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]

### AE
#### Words
<s> ### Return five key noun phrases, verbs, or adverbs that appear in this Bible verse. Only return phrases that are in the verse provided. ### Verse: <context> ### Key words:


#### Phrases
<s> ### Return five key phrases that appear in this Bible verse. Separate the phrases by commas. Only return phrases that are in the verse provided. ### Verse: <context> ### Key phrases:

#### Lists
Performance: decent at times
<s> ### Extract any lists that appear in this Bible verse. Number the lists and separate them by commas. If there are no lists return "no lists". Only return lists that are in the verse provided. ### Verse: <context> ### Lists:

#### All
Performance: decent enough to try <- current AE prompt
<s> ${delim} Extract any nouns, noun phrases, actions, key phrases, and lists that appear in the following context and return them separated by <sep>. ${delim} Verse: <context> ${delim} Nouns, noun phrases, actions, key phrases, and lists: <answer>

Performance: mid
<s>[INST] <<SYS>>\n{Extract any entities, actions, key phrases, or lists that appear in the following prompt. Separate each unit by a hashtag (#). Only return words, phrases, and lists that are in the prompt provided.}\n<</SYS>>\n\n{<context>} [/INST]

### QAG
Performance: mid
<s> ### Return several extractive question and answer pairs based on the following context. Each pair should consist of only the question and the correct answer. The questions should not be abstract or subjective. The answers should not be multiple choice. Only return words that are in the context provided. ### Verse: <context> ### Question answer pairs:

Performance: can't seem to stop itself but produces decent Q&A
<s> ### You are a question and answer generator. Write several extractive question and answer pairs based on the provided context. The questions should not be abstract or subjective, but rather based directly on the context.\n\nEach pair should have a question delimited by "Q:" on the first line and then an answer delimited by "A:" on the next line. Put an extra delimiting line between each question-answer pair. ### Context: <context> ### Question and answer pairs:

Performance: mid, can't consistently stop
<s>[INST] <<SYS>>\n{You are a question and answer generator. Write several extractive question and answer pairs based on the provided context. The questions should not be abstract or subjective, but rather based directly on the context.\n\nEach pair should have a question delimited by "Q:" on the first line and then an answer delimited by "A:" on the next line. Put an extra delimiting line between each question-answer pair.}\n<</SYS>>\n\n{<context>} [/INST]

 <!-- in a comma-separated list: -->