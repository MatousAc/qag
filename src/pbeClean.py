import pandas as pd
import numpy as np
import re

dataSrc = '../data/pbe/raw'
dataDest = '../data/pbe/clean/allQuestions.csv'

# get data, remove columns, set proper dtypes
# ['refQuestion', 'answer', 'categories', 'source', 'quality']
lsb = pd.read_excel(f'{dataSrc}/Reformatted Bible Questions.xlsx')
lsb['categories'] = lsb['categories'].str.lower()
# ['question', 'answer', 'points', 'book', 'chapter', 'verse', 'endVerse', 'source']
bab = pd.read_csv(f'{dataSrc}/bible-questions.csv', sep=';')
bab = bab[(bab['chapter'] == bab['endChapter']) | (bab['endChapter'].isnull())] # rm questions that span multiple chapters
bab = bab.drop(columns=['id', 'type', 'endBook', 'dateCreated', 'endChapter']) # unnecessary columns
bab['source'] = 'Babienco'; bab['quality'] = np.nan

# 1. trim whitespace, rm consecutive spaces
lsb = lsb.replace(r"^ +| +$", r"", regex=True).replace(r'\s+', ' ', regex=True)
bab = bab.replace(r"^ +| +$", r"", regex=True).replace(r'\s+', ' ', regex=True)
# 2. basic deduplicate on reference, question, and answer
lsb = lsb.drop_duplicates(subset=['refQuestion', 'answer'])
bab = bab.drop_duplicates(subset=['question', 'answer', 'book', 'chapter', 'verse', 'endVerse'])
# 3. filter to remove FITB && T/F
fitbRe = '_|fitb|fill in the blanks|t/f|true or false/i' # ignore case
lsb = lsb[lsb['refQuestion'].str.contains(fitbRe) == False]
bab = bab[bab['question'].str.contains(fitbRe) == False]
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
  lsb[cat] = lsb['categories'].str.contains(cat.lower())
  bab[cat] = np.nan
# 7. extract reference "according to..." && capitalize question
refRe = r'\s*According to (?P<book>(?:\d\s)?[a-zA-Z]+)\s(?P<chapter>\d+):(?P<verse>\d+)(?:[-,]?(?P<endVerse>\d+))?,?\s*'
newCols = lsb['refQuestion'].str.extract(refRe, flags=re.IGNORECASE)
lsb['question'] = lsb['refQuestion'].str.replace(refRe, '', flags=re.IGNORECASE, regex=True)
lsb['question'] = lsb['question'].str.slice(stop=1).str.capitalize() + lsb['question'].str.slice(start=1)
lsb = pd.concat([lsb, newCols], axis=1)
# 8. combine datasets
cols = ['book', 'chapter', 'verse', 'endVerse', 'question', 'answer', 'points', 'source', 'quality']
cols += cats
lsb = lsb[cols]
bab = bab[cols]
data = pd.concat([lsb, bab])
# 9. change column type as necessary
data['endVerse'] = data['endVerse'].fillna(data['verse'])
data['chapter'] = data['chapter'].astype(np.int64)
data['verse'] = data['verse'].astype(np.int64)
data['endVerse'] = data['endVerse'].astype(np.int64)
data['points'] = data['points'].astype(np.int64)
data['answer'] = data['answer'].astype(str)
# 10. format all numbered answers the same: (#)
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
# 11. uncapitalize unnecessarily capitalized words like "WHY", "WHAT", "WHICH", "NOT", "FROM"
capsRe = r'\b[A-Z]{2,}\b'
data['question'] = data['question'].str.replace(capsRe, lambda match: match.groups(1), regex=True)
# 12. remove surrounding quotes and periods
data['answer'] = data['answer'].str.replace(r'^"+|"+$|\.+$', r"", regex=True)
# 13. Remove Be Specific, any caps, w/ or without parentheses
data['question'] = data['question'].str.replace(r'\s*\(?Be Specific\)?\s*/i', r"", regex=True)
# 14. transform "v #" to "verse #"
data['question'] = data['question'].str.replace(r'v\s(\d+)', lambda match: f'verse {match.groups()[0]}', regex=True)
# 15. remove any rows with a point-value greater than 15
data = data[data['points'] < 13]
# 16. final deduplication based on reference, question, and answer (we lose about 500 questions here 👍)
data = data.drop_duplicates(subset=['book', 'chapter', 'verse', 'question', 'answer'])

# finally save
print(f'Saving this format to {dataDest}')
print(f'Data\nshape: {data.shape}\ncols:\n{data.dtypes}')
data.to_csv(dataDest, index=False)



# df = pd.DataFrame(np.array([
#   ['(1) Pontus (2) Galatia (3) Cappadocia (4) Asia (5) Bithynia', 5],
#   ['the holy, amazing, blessed blood of Jesus Christ', 1],
#   ['(1) grace and (2) peace', 1],
#   ['1. Came to Jerusalem 2. Besieged it (Jerusalem)', 1],
#   ['1) Jehoiakim, king of Judah 2) some of the articles of the house of God', 2],
#   ['1) changes time 2) seasons 3) kings 4) up kings 5)  wise 6) knowledge', 6],
#   ['1) iron 2)clay 3) bronze 4) silver 5.gold', 1],
#   ['You (Nebuchadnezzar)', 1],
#   ['1) the plain of Dura 2)the province of Babylon', 2],
#   ['A watcher, a holy one', 2],
#   ['Drive you; dwell beasts; eat grass; be wet', 1]
# ]), columns=['answer', 'points'])

