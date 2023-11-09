import pandas as pd
import numpy as np
import re

dataSrc = '../data/pbe/raw'
dataDest = '../data/pbe/clean'

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
questionCol = lsb['refQuestion'].str.replace(refRe, '', flags=re.IGNORECASE, regex=True)
lsb.insert(loc=0, column='question', value=questionCol)
lsb['question'] = lsb['question'].str.slice(stop=1).str.capitalize() + lsb['question'].str.slice(start=1)
lsb = pd.concat([lsb, newCols], axis=1)
# 8. combine datasets
cols = ['book', 'chapter', 'verse', 'endVerse', 'question', 'answer', 'points', 'source', 'quality']
cols += cats
lsb = lsb[cols]
bab = bab[cols]
data = pd.concat([lsb, bab])
print(f'Data\nshape: {data.shape}\ncols: {data.columns}')
# 9. format all numbered answers same (#)

print(data[['book', 'chapter', 'verse', 'endVerse', 'question', 'answer']])

# remove periods from the ends of answers
# remove quotes from the answer if they are the first and last characters
# uncapitalize unnecessarily capitalized words like "WHY", "WHAT", "WHICH", "NOT", "FROM"
# Remove Be Specific, any caps, w/ or without parentheses
# transform "v #" to "verse #"
# remove any rows with a point-value greater than 15
# print(bab[bab['points'] > 14])





# bab[bab['endVerse'].isnull()]['endVerse'] = bab['verse']
# bab['endVerse'] = bab['endVerse'].astype(np.int64)
# bab['verse'] = bab['verse'].astype(np.int64)
# s = pd.Series([
#   'According to John 12:13 . . .',
#   'According to 1 Kings 2:3 . . .',
#   'According to 1 Peter 1:2, v 2? (6000 points)'
# ])
# ref = s.str.extract(refRe, flags=re.IGNORECASE)
# s = s.str.replace(refRe, '', flags=re.IGNORECASE, regex=True)
# print(ref)
# print(s)