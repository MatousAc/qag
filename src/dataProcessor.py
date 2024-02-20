import sys, pandas as pd, random, csv, json
from datasets import load_dataset, Dataset
from qagBase import QAGBase

class DataProcessor(QAGBase):
  def configure(self):
    self.source = self.paths['dpSource']
    self.destination = self.paths['dpDest']
    
    self.nkjv = pd.read_csv(self.paths['dpBibleSrc'])
    self.nkjvInfo = self.getNkjvInfo()


  # creates an object representing the NKJV Bible
  def getNkjvInfo(self):
    nkjvContent = {}

    with open(self.paths['dpBibleSrc'], 'r', encoding='utf-8') as csvFile:
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
    data = pd.read_csv(self.source)
    data['sentence'] = ''; data['paragraph'] = ''; data['paragraph_question'] = ''; data['paragraph_sentence'] = ''
    if not self.quiet: print(f'dataset length: {len(data.index)}')

    maximum = len(data.index)
    for i, row in data.iterrows():
      # show progress
      self.printProgressBar(i, maximum = maximum, label = 'context', width = 75)
      # get verse pieces
      book = row['book']; chapter = row['chapter']; verse = row['verse']; end = row['endVerse']
      previousNum = verse - 1 if verse > 1 else None
      followingNum = end + 1 if self.nkjvInfo[book][chapter] > end else None
      previous = self.getVerse(book, chapter, previousNum) + ' ' if previousNum else ''
      sentence = self.getVerse(book, chapter, verse, endVerse=end)
      following = ' ' + self.getVerse(book, chapter, followingNum) if followingNum else ''
      # assign pieces
      data.at[i, 'sentence'] = sentence
      data.at[i, 'paragraph'] = previous + sentence + following
      data.at[i, 'paragraph_question'] = f'question: {row["question"]}, context: {previous + sentence + following}'
      data.at[i, 'paragraph_sentence'] = f'{previous}<hl> {sentence} <hl>{following}'
    self.printProgressBar(maximum, maximum = maximum, label = 'context', width = 75) # done
    print('\n')
    # reorganize columns
    cols = ['answer', 'paragraph_question', 'question', 'sentence', 'paragraph', 'paragraph_sentence', 'points', 'source', 'quality']
    data = data[cols]
    if not self.quiet: print(data.head())
    # drop rows that we can't get references for
    data = data.dropna(subset=['paragraph_question', 'sentence', 'paragraph', 'paragraph_sentence'])
    # save
    data.to_csv(self.destination, index=False)
  
  def makeAE(self):
    print(self.source)
    # load json of csv
    if ".csv" not in self.source:
      dataset = load_dataset(self.source)
      df = dataset['train'].to_pandas()
    else: df = pd.read_csv(self.source)

    df = df[['answer', 'question','sentence']]
    grouped = df.groupby('sentence').agg({
      'answer': lambda x: ' <sep> '.join(set(x)), 
      'question': 'count'
    }).reset_index()
    grouped.rename(columns={'question': 'count'}, inplace=True)
    if not self.quiet: print(f'Mean Q&A per context: {grouped["count"].mean()}')
    if not self.quiet: print(grouped.head())
    if not self.quiet: print(len(grouped))
    dataset = Dataset.from_pandas(grouped.reset_index(drop=True))
    if not self.quiet: print(dataset)
    dataset.to_json(f'{self.destination}/data.json')
    
  def csvToJsonl(self):
    pd.read_csv(self.source).to_json(self.destination, orient='records', lines = True)
    
  def jsonlToCsv(self):
    with open(self.source) as f:
      lines = f.read().splitlines()
    data = pd.DataFrame(lines)
    data.columns = ['json_element']
    data = pd.json_normalize(data['json_element'].apply(json.loads))
    data.to_csv(self.destination)

  def getVerse(self, book, chapter, startVerse, endVerse = None):
    # default to start verse
    endVerse = endVerse if endVerse else startVerse

    result = ''
    for verseNumber in range(startVerse, endVerse + 1):
      result += self.nkjv.loc[
        (self.nkjv['verseNumber'] == verseNumber)
        & (self.nkjv['book'] == book)
        & (self.nkjv['chapterNumber'] == chapter)
      , 'verse'].values[0]
    return result

  def getRandomVerse(self): 
    book = random.choice(self.nkjv['book'].unique())
    df = self.nkjv.loc[self.nkjv['book'] == book]
    chapter = random.choice(df['chapterNumber'].unique())
    df = df.loc[df['chapterNumber'] == chapter]
    verse = random.choice(df['verseNumber'].unique())
    return df.loc[df['verseNumber'] == verse].values[0][-1]

if __name__ == '__main__':
  dp = DataProcessor()
  match sys.argv[1].replace('-', '').lower():
    case 'randomverse': print(dp.getRandomVerse())
    case 'makeae': dp.makeAE()
    case 'pbecontextualize': dp.pbeContextualize()
    case 'csvtojsonl': dp.csvToJsonl()
    case 'jsonltocsv': dp.jsonlToCsv()
    case 'none' | _: pass


