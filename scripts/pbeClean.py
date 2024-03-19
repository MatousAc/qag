# to start out, place the huge 'Reformatted Bible Questions.xlsx'
# && Babienco's 'bible-questions.csv' in /data/pbe/raw
import pandas as pd, numpy as np, re, csv

dataSrc = '../data/pbe/raw'
dataDest = '../data/pbe/clean/refQuestions.csv'

# get data, remove columns, set proper dtypes
# ['refQuestion', 'answer', 'categories', 'source', 'quality']
lsb = pd.read_excel(f'{dataSrc}/Reformatted Bible Questions.xlsx')
lsb['categories'] = lsb['categories'].str.lower()
# ['question', 'answer', 'points', 'book', 'chapter', 'verse', 'endVerse', 'source']
bab = pd.read_csv(f'{dataSrc}/bible-questions.csv', sep=';')
bab = bab[(bab['chapter'] == bab['endChapter']) | (bab['endChapter'].isnull())] # rm questions that span multiple chapters
bab = bab.drop(columns=['id', 'type', 'endBook', 'dateCreated', 'endChapter']) # unnecessary columns
bab['source'] = 'Babienco'; bab['quality'] = 8;

# 1. trim whitespace, rm consecutive spaces
lsb = lsb.replace(r'^ +| +$', r'', regex=True).replace(r'\s+', ' ', regex=True)
bab = bab.replace(r'^ +| +$', r'', regex=True).replace(r'\s+', ' ', regex=True)
# 2. basic deduplicate on reference, question, and answer
lsb = lsb.drop_duplicates(subset=['refQuestion', 'answer'])
bab = bab.drop_duplicates(subset=['question', 'answer', 'book', 'chapter', 'verse', 'endVerse'])
# 3. filter to remove FITB && T/F
fitbTF = r'_|fitb|fill in the blanks|t\/f|true or false'
lsb = lsb[lsb['refQuestion'].str.contains(fitbTF, regex=True, flags=re.IGNORECASE) == False]
bab = bab[bab['question'].str.contains(fitbTF, regex=True, flags=re.IGNORECASE) == False]
ansTF = r'^(?:true|false)'
lsb = lsb[lsb['answer'].str.contains(ansTF, regex=True, flags=re.IGNORECASE) == False]
bab = bab[bab['answer'].str.contains(ansTF, regex=True, flags=re.IGNORECASE) == False]
# 4. drop rows with missing references, questions, or answers
lsb = lsb.dropna(subset=['refQuestion', 'answer'])
bab = bab.dropna(subset=['question', 'answer', 'book', 'chapter', 'verse'])
# 5. extract point values "2 points", "2-pts." etc.
ptsRe = r'\s*\(?(?P<points>\d+)\s*-?(?:point|pt|pt)s?\.?\)?\s*'
lsb['refQuestionCategories'] = lsb['refQuestion'] + lsb['categories'].astype(str)
lsb['points'] = lsb['refQuestionCategories'].str.extract(ptsRe, flags=re.IGNORECASE)
lsb['points'] = lsb['points'].replace(np.nan, 1).astype(np.int64)
lsb['refQuestion'] = lsb['refQuestion'].str.replace(ptsRe, '', flags=re.IGNORECASE, regex=True)
lsb['categories'] = lsb['categories'].str.replace(ptsRe, '', flags=re.IGNORECASE, regex=True)
# 6. pull categories out into their own columns
cats = ['2To3', 'bigPoints', 'people', 'places', 'names', 'numbers']
for cat in cats:
  lsb[cat] = lsb['categories'].str.contains(cat.lower()) == True
  bab[cat] = False # Babienco has no categories
# 7. extract reference "according to..." && capitalize question
refRe = r'\s*According to (?P<book>(?:\d\s)?[a-zA-Z]+)\s(?P<chapter>\d+):(?P<verse>\d+)(?:[-,]?[ ]?(?P<endVerse>\d+))?,?\s*'
newCols = lsb['refQuestion'].str.extract(refRe, flags=re.IGNORECASE)
lsb['question'] = lsb['refQuestion'].str.replace(refRe, '', flags=re.IGNORECASE, regex=True)
lsb['question'] = lsb['question'].str.slice(stop=1).str.capitalize() + lsb['question'].str.slice(start=1)
lsb = pd.concat([lsb, newCols], axis=1)
# 8. combine datasets
cols = ['book', 'chapter', 'verse', 'endVerse', 'question', 'answer', 'points', 'source', 'quality']
cols += cats
lsb = lsb[cols]
bab = bab[cols]
# putting lsb first means that their "higher quality" questions will be chosen
# as the row to stay when deduplicating
data = pd.concat([lsb, bab])
# 9. change column type as necessary
data['endVerse'] = data['endVerse'].fillna(data['verse'])
data['chapter'] = data['chapter'].astype(np.int64)
data['verse'] = data['verse'].astype(np.int64)
data['endVerse'] = data['endVerse'].astype(np.int64)
data['points'] = data['points'].astype(np.int64)
data['answer'] = data['answer'].astype(str)
# 10. drop rows where endVerse was parsed as larger than start verse
data = data[data['endVerse'] >= data['verse']]
# 11. format all numbered answers the same: (#)
def formatAnswers(answer: str, allegedPoints: int):
  # define f(x) to process each number
  def processAnswer(match):
    number = next(group for group in match.groups() if group is not None)
    return f'({number}) ' # format
  
  numRe = r'(?:(?:\((\d+)\))|(?:(\d+)\.)|(?:(\d+)\)))\s*'
  # sub different formats w/ correct format
  answer = re.sub(numRe, processAnswer, answer)
  pointCount = countPoints(answer)
  
  if (pointCount == 1):
    parts = answer.split(';') # semicolons
    # only split on commas if multiple points reported
    if (len(parts) == 1) and (allegedPoints > 1):
      parts = answer.split(',')
    if len(parts) > 1:
      res = []
      currentNumber = 1
      for part in parts:
        res.append(f'({currentNumber}) {part.strip()}')
        currentNumber += 1
      answer = ' '.join(res)
  return answer

def countPoints(answer: str):
  numRe = r'\((\d+)\)'
  return max(len(re.findall(numRe, answer)), 1)

data['answer'] = data.apply(lambda row: formatAnswers(row['answer'], row['points']), axis=1)
data['points'] = data['answer'].apply(countPoints)
data['2To3'] = (data['points'] > 1) & (data['points'] <= 3)
data['bigPoints'] = data['points'] > 3
# 12. uncapitalize unnecessarily capitalized words like "WHY", "WHAT", "WHICH", "NOT", "FROM", "TRUE", "FALSE"
capsRe = r'\b([A-Z]{2,})\b'
data['question'] = data['question'].str.replace(capsRe, lambda match: match.groups()[0].lower(), regex=True)
data['answer'] = data['answer'].str.replace(capsRe, lambda match: match.groups()[0].lower(), regex=True)
# 13. remove Be Specific, any caps, w/ or without parentheses, periods && various misspellingsc
data['question'] = data['question'].str.replace(r'\s*\(?be\sspe((cific)|(ific)|(cfic)|(cifc))\.?\)?\.?\s*', r'', flags = re.IGNORECASE, regex=True)
# 14. transform "v #" to "verse #"
data['question'] = data['question'].str.replace(r'v\s(\d+)', lambda match: f'verse {match.groups()[0]}', regex=True)
# 15. remove any rows with a point-value greater than 13
data = data[data['points'] < 13]
# 16. put back apostrophes that somehow got lost
data['question'] = data['question'].str.replace('�', "'")
data['answer'] = data['answer'].str.replace('�', "'")
# 17. remove all occurences of "Do not confuse with verse #" in question and answer columns
data['question'] = data['question'].str.replace(r'\(?do not confuse .*\)?\.?', r'', regex=True, flags=re.IGNORECASE)
data['answer'] = data['answer'].str.replace(r'\(?do not confuse .*\)?\.?', r'', regex=True, flags=re.IGNORECASE)
# 18. replace special/double characters
data['answer'] = data['answer'].str.replace(r'“|”', r'"', regex=True).str.replace(r"‘|’", r"'", regex=True)
data['question'] = data['question'].str.replace(r'“|”', r'"', regex=True).str.replace(r"‘|’", r"'", regex=True)
# 19. remove surrounding quotes and periods
data['answer'] = data['answer'].str.replace(r'^"(.+)"$', r'\1', regex=True).str.replace(r"^'(.+)'$", r"\1", regex=True)
data['question'] = data['question'].str.replace(r'^"(.+)"$', r'\1', regex=True).str.replace(r"^'(.+)'$", r"\1", regex=True)
# 20 remove anything within parentheses that is not just a point value
data['answer'] = data['answer'].str.replace(r'\(\d*[a-zA-Z\s\.?!\-\'"]+\d*\)', r'', regex=True)
data['question'] = data['question'].str.replace(r'\(\d*[a-zA-Z\s\.?!\-\'"]+\d*\)', r'', regex=True)
# 21. final trimming and stripping
data['answer'] = data['answer'].str.strip().str.strip('.')
data['question'] = data['question'].str.strip()
# 22. replace "None" with "none" so that we can keep these values next time we load them as csv
data['answer'] = data['answer'].str.replace(r'^None$', r'^none$', regex = True)
# 23. final deduplication based on reference, question, and answer (we lose about 500 questions here 👍)
data = data.drop_duplicates(
  keep='first', # lsb generally has higher quality and we want to keep that designation
  subset=['book', 'chapter', 'verse', 'question', 'answer']
)


# finally save
data.to_csv(dataDest, index=False) # to get rid of double quotes: quoting=csv.QUOTE_NONE, escapechar="\\"
print(f'Data\nshape: {data.shape}\ncols:\n{data.dtypes}')


# df = pd.DataFrame(np.array([
#   ['(1) Pontus (2) Galatia (3) Cappadocia (4) Asia (5) Bithynia', 5],
#   ['the holy, amazing, blessed BLOOD of Jesus Christ', 1],
#   ['(1) GRACE and (2) peace', 1],
#   ['1. Came to Jerusalem 2. Besieged it (Jerusalem)', 1],
#   ['1) Jehoiakim, king of Judah 2) some of the articles of the house of God', 2],
#   ['1) changes time 2) seasons 3) kings 4) up kings 5)  wise 6) knowledge', 6],
#   ['1) iron 2)clay 3) BRONZE 4) silver 5.gold', 1],
#   ['You (Nebuchadnezzar)', 1],
#   ['1) the plain of Dura 2)the province of Babylon', 2],
#   ['A watcher, a HOLY ONE', 2],
#   ['Drive you; dwell beasts; eat grass; be wet', 1]
# ]), columns=['question', 'points'])

# df['question'] = df['question'].str.replace(capsRe, lambda match: match.groups()[0].lower(), regex=True)

# print(df['question'])