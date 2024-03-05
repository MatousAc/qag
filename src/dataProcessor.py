import pandas as pd, sys, random, csv, json, string, re
from datasets import load_dataset, Dataset
from qagBase import QAGBase
from verse import Verse

class DataProcessor(QAGBase):
  def configure(self):
    self.source = self.paths['dpSource']
    self.destination = self.paths['dpDest']
    self.bibleSrc = self.basePath + '/data/bible/nkjv.csv'
    
    self.nkjv = pd.read_csv(self.bibleSrc)
    self.nkjvInfo = self.getNkjvInfo()

  def getNkjvInfo(self):
    '''Creates an object representing the NKJV Bible'''
    nkjvContent = {}

    with open(self.bibleSrc, 'r', encoding='utf-8') as csvFile:
      reader = csv.reader(csvFile)
      next(reader)  # skip header row

      for row in reader:
        book, chapter, verse, text = row
        chapter = int(chapter)
        verse = int(verse)

        if book not in nkjvContent:
          nkjvContent[book] = {'numChapters': 0}

        if chapter > nkjvContent[book]['numChapters']:
          nkjvContent[book]['numChapters'] = chapter

        if chapter not in nkjvContent[book]:
          nkjvContent[book][chapter] = 0

        nkjvContent[book][chapter] = max(nkjvContent[book][chapter], verse)

    return nkjvContent

  def pbeContextualize(self):
    '''Adds the actual verse texts to data'''
    data = pd.read_csv(self.source)
    data['sentence'] = ''; data['paragraph'] = ''; data['paragraph_question'] = ''; data['paragraph_sentence'] = ''
    if not self.quiet: print(f'dataset length: {len(data.index)}')

    maximum = len(data.index)
    for i, row in data.iterrows():
      # show progress
      self.printProgressBar(i, maximum = maximum, label = 'context')
      # get verse pieces
      book = row['book']; chapter = row['chapter']; start = row['verse']; end = row['endVerse']
      verse: Verse = self.constructVerse(book, chapter, start, end)
      # assign pieces
      data.at[i, 'sentence'] = verse.text
      data.at[i, 'paragraph'] = verse.inContext
      data.at[i, 'paragraph_question'] = f'question: {row["question"]}, context: {verse.inContext}'
      data.at[i, 'paragraph_sentence'] = f'{verse.previous}<hl> {verse.text} <hl>{verse.following}'
    self.printProgressBar(maximum, maximum = maximum, label = 'context') # done
    print('\n')
    # reorganize columns
    cols = ['answer', 'paragraph_question', 'question', 'sentence', 'paragraph', 'paragraph_sentence', 'points', 'source', 'quality']
    data = data[cols]
    if not self.quiet: print(data.head())
    # drop rows that we can't get references for
    data = data.dropna(subset=['paragraph_question', 'sentence', 'paragraph', 'paragraph_sentence'])
    # drop rows that have sentences w/ more than 150 words
    data = data[data["sentence"].apply(lambda x: len(x.split()) <= 150)]
    # save
    data.to_csv(self.destination, index=False)

  def aeDeduplicate(self, answers):
    '''Set-based deduplication of similar answers'''
    def normAns(text):
      text = text.lower()
      text = re.sub(r'\((\d{1,3})\) ', '', text) # point marker removal
      text = text.translate(str.maketrans("", "", string.punctuation))
      return text
    
    def areSimilar(text1, text2):
      # Split texts into words
      words1 = set(normAns(text1).split())
      words2 = set(normAns(text2).split())
      len1 = len(words1); len2 = len(words2)
      maxLen = max(len1, len2); minLen = min(len1, len2)
      # answers are not similar if they are more than 3 words apart
      # or if the difference is larger than 40% of the smaller answer
      if abs(len1 - len2) > max(int(minLen * 0.4), 3): return False
      # answers are similar if they intersect 70% of the time or only
      # differ by one word for smaller answers
      threshold = max(maxLen - 1, 1) if maxLen < 5 else int(0.7 * maxLen)
      return len(words1.intersection(words2)) >= threshold

    # rm near or full duplicates
    uniqueElems = set()
    for elem in answers:
        isDuplicate = any(areSimilar(elem, uniqueElem) for uniqueElem in uniqueElems)
        if not isDuplicate: uniqueElems.add(elem)
    return uniqueElems

  def makeAE(self):
    '''Aggregates verses to get AE data'''
    # load json of csv
    if ".csv" not in self.source:
      dataset = load_dataset(self.source)
      df = dataset['train'].to_pandas()
    else: df = pd.read_csv(self.source)
    df = df[['answer', 'question','sentence', 'quality']]

    # group answers by verse and remove near duplicates
    sep = ' <sep> '
    grouped = df.groupby('sentence').agg({
      'answer': lambda x: sep.join(self.aeDeduplicate(x)),
      'quality': 'mean'
    }).reset_index()
    grouped['count'] = grouped['answer'].apply(lambda x: x.count(sep) + 1)
    grouped.rename(columns={'question': 'count'}, inplace=True)
    grouped = grouped[grouped['count'] > 2] # train model to produce 3+ ans
    dataset = Dataset.from_pandas(grouped.reset_index(drop=True))
    if not self.quiet: 
      print(grouped.head())
      print(len(dataset))
      print(f'Mean Q&A per context: {round(grouped["count"].mean(), 2)}')
    dataset.to_json(self.destination)
    
  def csvToJsonl(self):
    '''Converts CSV to JSONL'''
    pd.read_csv(self.source).to_json(self.destination, orient='records', lines = True)
    
  def jsonlToCsv(self):
    '''Converts JSONL to CSV'''
    with open(self.source) as f:
      lines = f.read().splitlines()
    data = pd.DataFrame(lines)
    data.columns = ['json_element']
    data = pd.json_normalize(data['json_element'].apply(json.loads))
    data.to_csv(self.destination)

  def constructVerse(self, *args) -> Verse:
    '''Returns a Verse object given a reference.
    Reference can be a normally formatted string such as John 3:16-17,
    or a book, chapter, and verse start/end numbers.'''
    # let the constructor handle lots of the setup
    v = Verse(*args)
    # get context verse numbers if applicable
    previousNum = v.start - 1 if v.start > 1 else None
    followingNum = v.end + 1 if self.nkjvInfo[v.book][v.chapter] > v.end else None

    # get verse text
    chapter = self.nkjv.loc[
      (self.nkjv['book'] == v.book)
      & (self.nkjv['chapterNumber'] == v.chapter)
    ]
    def getVrs(num: int): return chapter.loc[chapter['verseNumber'] == num, 'verse'].values[0]
    
    v.previous = getVrs(previousNum) if previousNum else ''
    v.following = getVrs(followingNum) if followingNum else ''
    
    targetVerses = []
    for verseNumber in range(v.start, v.end + 1):
      targetVerses.append(getVrs(verseNumber))
    v.text = ' '.join(targetVerses)
    v.inContext = f'{v.previous} {v.text} {v.following}'.strip()
    return v

  def getRandomVerse(self) -> Verse: 
    '''Returns a Verse object based on a random selection of
    book, chapter, and verse number.'''
    book = random.choice(self.nkjv['book'].unique())
    df = self.nkjv.loc[self.nkjv['book'] == book]
    chapter = random.choice(df['chapterNumber'].unique())
    df = df.loc[df['chapterNumber'] == chapter]
    verseNumber = random.choice(df['verseNumber'].unique())
    return self.constructVerse(book, chapter, verseNumber)

if __name__ == '__main__':
  dp = DataProcessor()
  match sys.argv[1].replace('-', '').lower():
    case 'randomverse': print(dp.getRandomVerse().text)
    case 'makeae': dp.makeAE()
    case 'pbecontextualize': dp.pbeContextualize()
    case 'csvtojsonl': dp.csvToJsonl()
    case 'jsonltocsv': dp.jsonlToCsv()
    case 'none' | _: pass


