import sys, pandas as pd, random
from datasets import load_dataset, Dataset
from qagBase import QAGBase

class DataProcessor(QAGBase):
  def configure(self):
    self.dpCf = self.cp['dataProcessor']
    self.source = self.dpCf['dpSource']
    self.destination = self.dpCf['dpDest']
    
    bibleDataSource = '../data/bible/nkjv.csv'
    self.nkjv = pd.read_csv(bibleDataSource)

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
    print(f'Ephesians 2:10 - {self.getVerse("Ephesians", 2, 10)}')


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
  match sys.argv[1]:
    case 'randomVerse' | '-randomVerse': print(dp.getRandomVerse())
    case 'qgToAE' | '-qgToAE': dp.qgToAE()
    case 'pbeContextualize' | '-pbeContextualize': dp.pbeContextualize()
    case 'none' | _: pass


