# Trained model notes
The best models so far are emboldened
## AE
* 7b-chatAE00: Specific prompt. No custom tokens. Produces relatively good extracted answers, but sometimes duplicated ones. Does not stop short.
* 7b-chatAE01: Specific prompt. No custom tokens. Further de-duplicated data. Produces decent answers w/ less duplication. Does not stop short.
* 7b-chatAE02: Specific prompt. Custom tokens. Latest data w/o the "Do not confuse with . . ." Produces trash. Does not stop short.
* 7b-chatAE03: Less specific prompt. No custom tokens. Produces good answers. Does not stop short.
* 7b-chatAE04: Unspecific prompt. No custom tokens. Trained on max data. Does just as well as above.
* 7b-chatAE05: Unspecific prompt. No custom tokens. Testing cutom eval metrics. The generation seems to get slightly better as we go along through training, but the metrics don't change as much as I'd expect. The end result extracts good answers.
* 7b-chatAE06: Same as above. Testing training more Lora layers (qvko). Results: the adapter is twice the size of the others at 256MB. Inference is not slower however. The answers produced can be slightly better than the 07 model which did not train extra layers. These answers are slightly easier to make questions for. Probably move forward without the 'o' layer, but keep training the 'k' (key layer).
* 7b-chatAE07: Same as above. Testing exact same w/o more Lora layers (increased gradient accumulation steps to 10). Produces slightly more difficult answers than the 06 model which has trained extra layers.
* 7b-chatAE08: Same as above. Now testing w/ injection of custom evaluation metrics. Results are same as above.
* 7b-chatAE09: New data (Paulean && NAD). Train w/ optimal hyperparameters. Quality threshold 8. Inference is marginally slower potentially due to more layers? Answers generated are pretty good. Some slight improvement possibly. The new deduplication script is helping.
* 7b-chatAE10: Training with newly deduplicated && point/length filtered AE data. This model generally produces less multi-point questions, but usually when it does, the multiple points are slightly more relevant and easier to craft questions for. This isn't strictly true, but on average, yes they are better. There may be something to explore here, but I can't mark this as the best model just yet.
* **7b-chatAE11**: Training with new data. Not filtering, just deduplicating. Produces less multi-point answers. Some multi-point answers are connected properly, but some are not. Overall, does a bit better with single-point questions and a tiny bit better with multi-point questions. One thing I should probably apply in the future is the "long answer" filter, as this model is more likely to produce longer answers. We could also revert to the 09 AE version. It's close to as good.

## QG
* 7b-chatQG00: Relatively specific unoptimized prompt. No custom tokens. Produces decent questions. Spits out a ton of "do not confuse w/ vs #"
* 7b-chatQG01: Relatively specific unoptimized prompt. No custom tokens. Removed "do not confuse" from training data. Produces good questions, though they are a bit short. Spits out trash after the question.
* 7b-chatQG02: Relatively specific unoptimized prompt. Custom tokens. Produces decent questions, though those that are longer are too long. Spits out trash after the question.
* 7b-chatQG03: Unspecific prompt. No custom tokens. Produces good questions 80% of the time but then continues with trash.
* 7b-chatQG04: Unspecific prompt. No custom tokens. Trained on max amount of data. Produces more relevant, correct, and sensible questions than trainings with partial data. Definitely a bit better than before. Also produces less trash than before.
* 7b-chatQG05: Testing w/o more Lora layers but with quality threshold 9. Produces almost no trash, but does produce several questions for each answer. The first question's quality is quite good, but the model makes things up for short contexts. Adding extra context for short verses on just QG ($< 15$ words) improves the performance quite a bit, so we'll be doing that from now on.
* 7b-chatQG06: Testing w Lora layers (qvk) and quality threshold 9. Results seem to be like the above. Questions are seem relevant at first and the model even seems to correct itself in places, but in QAG this model underperforms. A significant amount of questions are off.
* 7b-chatQG07: Like 05 but now with the paragraph_sentence context. When prompted without extra context and hl tokens, the model still seemed to produce good questions. It produces more trash and newlines afterwards though. When prompted with the extra context as trained, the model produces decent questions. It significantly improves QAG for very short verses. It seems to flounder less in general. The main downside is that it will sometimes guess at information that is not provided to it and will misplace a text in the Bible. This causes the model to sometimes produce completely wrong questions for the context. Also the model has not been discouraged from creativity enough, so it sometimes just makes up facts that are not found in the context or extended context. It is a major problem and we will not train the model this way in the future.
* 7b-chatQG08: New data (Paulean && NAD). More Lora layers and other optimal hyperparameters. Quality threshold 8. The gestion generation is significantly better. The questions are more articulate and exact. There is still a slight issue with potentially making stuff up from the wrong part of the bible. I should try training w/ references. There is a small potential for maybe training the smaller verses with the highlight tokens, as the model (on rare occasion) may mistake another part of context referring to the answer and then will mention stuff outside the context verse.
* 7b-chatQG09: Messed up training. Disregard.
* **7b-chatQG10**: Trained with references included. Also including the new Joshua and Judges data. Quality 8. Training ended with a broken pipe error. The QG seems to be the best so far. Although not perfect, it usually generates a coherent and relevant question with enough context. Adding in the reference seems to fix the problem of assuming the wrong information from the Bible from the QG side (though it does not fix the AE part doing the same). Continuing with this method henceforth.

# File Evaluators
* 0: Gabriella Grundy
* 1: Ryan Ramirez
* 2: Daryl Illangovan
* 3: Lisa Myaing
* 4: Beth deFlutier
* 5-7: Ki Song
* 8: Cheryl Craven
* 9: Michael Babienco
* 10: Ted Ashton
* 11: Emily Hamstra


# Input format testing
the \<s> at the start of each fo the following should not actually have the '\\'
## Text-completion models 
### AE
\<s> ### Here is a context verse. ### Verse: <context> ### Here are seven nouns and noun phrases in a comma-separated list that appear in this Bible verse:

## Chat Models
Examples of how to format:
* https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py
* https://github.com/facebookresearch/llama/issues/481

I won't actually format it like that because I don't need a chat-like response.

\<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]

### AE
#### Words
\<s> ### Return five key noun phrases, verbs, or adverbs that appear in this Bible verse. Only return phrases that are in the verse provided. ### Verse: <context> ### Key words:


#### Phrases
\<s> ### Return five key phrases that appear in this Bible verse. Separate the phrases by commas. Only return phrases that are in the verse provided. ### Verse: <context> ### Key phrases:

#### Lists
Performance: decent at times
\<s> ### Extract any lists that appear in this Bible verse. Number the lists and separate them by commas. If there are no lists return "no lists". Only return lists that are in the verse provided. ### Verse: <context> ### Lists:

#### All
Best specific optimized prompt. Performance: good.

\<s> ### Extract any nouns, noun phrases, actions, key phrases, and lists that appear in the following context and return them separated by <sep>. ### Verse: <context> ### Nouns, noun phrases, actions, key phrases, and lists: <answer>

Best overall. Unspecific prompt. Performance good. <- current prompt

\<s> ### Extract potential answers to questions and return them separated by <sep>. ### Verse: <context> ### Potential answers: <answer>

Performance: mid

\<s>[INST] <<SYS>>\n{Extract any entities, actions, key phrases, or lists that appear in the following prompt. Separate each unit by a hashtag (#). Only return words, phrases, and lists that are in the prompt provided.}\n<</SYS>>\n\n{<context>} [/INST]

### QG
Best specific unoptimized prompt. Performance: good

\<s> ### Given the following context verse and answer, write a question for the answer. ### Verse: <context> ### Answer: <answer> ### Question: <question>

Best unspecific prompt. Performance: good

\<s> ### Write a question for the context and answer. ### Verse: <context> ### Answer: <answer> ### Question: <question>

### QAG
Performance: mid

\<s> ### Return several extractive question and answer pairs based on the following context. Each pair should consist of only the question and the correct answer. The questions should not be abstract or subjective. The answers should not be multiple choice. Only return words that are in the context provided. ### Verse: <context> ### Question answer pairs:

Performance: can't seem to stop itself but produces decent Q&A

\<s> ### You are a question and answer generator. Write several extractive question and answer pairs based on the provided context. The questions should not be abstract or subjective, but rather based directly on the context.\n\nEach pair should have a question delimited by "Q:" on the first line and then an answer delimited by "A:" on the next line. Put an extra delimiting line between each question-answer pair. ### Context: <context> ### Question and answer pairs:

Performance: mid, can't consistently stop

\<s>[INST] <<SYS>>\n{You are a question and answer generator. Write several extractive question and answer pairs based on the provided context. The questions should not be abstract or subjective, but rather based directly on the context.\n\nEach pair should have a question delimited by "Q:" on the first line and then an answer delimited by "A:" on the next line. Put an extra delimiting line between each question-answer pair.}\n<</SYS>>\n\n{<context>} [/INST]

 <!-- in a comma-separated list: -->