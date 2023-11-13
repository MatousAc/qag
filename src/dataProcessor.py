import sys, pandas as pd, random, csv
from datasets import load_dataset, Dataset
from qagBase import QAGBase

class DataProcessor(QAGBase):
  def configure(self):
    self.dpCf = self.cp['dataProcessor']
    self.source = self.dpCf['dpSource']
    self.destination = self.dpCf['dpDest']
    
    bibleDataSource = '../data/bible/nkjv.csv'
    self.nkjv = pd.read_csv(bibleDataSource)

  # creates an object representing the NKJV Bible
  # should become a member in construction
  def getNkjvInfo(self):
    filePath = f'{self.dpCf["qagData"]}/bible/nkjv.csv'
    nkjvContent = {}

    with open(filePath, 'r', encoding='utf-8') as csvFile:
      reader = csv.reader(csvFile)
      next(reader)  # Skip header row

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

  def qgToAE(self):
    dataset = load_dataset(self.source)
    df = dataset['train'].to_pandas()
    df = df[['answer', 'question','sentence']]
    grouped = df.groupby('sentence').agg({
      'answer': lambda x: ' <sep> '.join(set(x)), 
      'question': 'count'
    }).reset_index()
    grouped.rename(columns={'question': 'count'}, inplace=True)
    print(f'Mean Q&A per context: {grouped["count"].mean()}')
    print(grouped.head())
    print(len(grouped))
    dataset = Dataset.from_pandas(grouped.reset_index(drop=True))
    print(dataset)
    dataset.to_json(f"{self.destination}/data.jsonl")
    
  def pbeContextualize(self):
    nkjvContent = self.getNkjvInfo()
    print(nkjvContent['Genesis']['numChapters'])
    print(nkjvContent['Genesis'][2])

    data = pd.read_csv(f'{self.source}/refQuestions.csv')
    data['sentence'] = ''; data['paragraph'] = ''; data['paragraph_question'] = ''; data['paragraph_sentence'] = ''

    print(data.head())
    # data['sentence'] = data.apply(lambda row: self.getVerse(row['book'], row['chapter'], row['verse'], endVerse=row['endVerse']), axis=1)
    print(len(data.index))
    for i in range(len(data.index)):
      row = data.iloc[i]

    #   print(f'{row["book"]} {row["chapter"]}:{row["verse"]}: {self.getVerse(row["book"], row["chapter"], row["verse"], endVerse=row["endVerse"])}')
    
    print(data.head())


  def getVerse(self, book, startChapter, startVerse, endChapter = None, endVerse = None):
    # default to start positions
    endChapter = endChapter if endChapter else startChapter
    endVerse = endVerse if endVerse else startVerse

    text = ""
    BookDf = self.nkjv[self.nkjv["Book"] == book]
    for index, row in BookDf.iterrows():
      chapter = row['ChapterNumber']
      verse = row['VerseNumber']
      if (chapter == startChapter and verse >= startVerse) or (chapter > startChapter):
        if chapter == endChapter and verse > endVerse:
          break
        text += row['Verse'] + " "
    return text

  def getRandomVerse(self):
    book = random.choice(self.nkjv['Book'].unique())
    df = self.nkjv.loc[self.nkjv['Book'] == book]
    chapter = random.choice(df['ChapterNumber'].unique())
    df = df.loc[df['ChapterNumber'] == chapter]
    verse = random.choice(df['VerseNumber'].unique())
    return df.loc[df['VerseNumber'] == verse].values[0][-1]

if __name__ == '__main__':
  dp = DataProcessor()
  match sys.argv[1].replace('-', ''):
    case 'randomVerse': print(dp.getRandomVerse())
    case 'qgToAE': dp.qgToAE()
    case 'pbeContextualize': dp.pbeContextualize()
    case 'none' | _: pass


