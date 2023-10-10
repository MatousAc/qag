import configparser
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import random

class DataProcessor():
  def __init__(self, configFilePath = 'qag.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.source = config['paths']['dpSource']
    self.destination = config['paths']['dpDest']
    self.genCf = config['general']
    self.quiet = self.genCf['quiet'] == 'True'
    self.dpCf = config['dataProcessor']

    bibleDataSource = '../data/bible/nkjv.csv'
    self.nkjv = pd.read_csv(bibleDataSource)

  def process(self):
    match self.dpCf['mode']:
      case 'none': return
      case 'randomVerse': print(dp.getRandomVerse())
      case 'qgToAE': self.qgToAE()

  def qgToAE(self):
    dataset = load_dataset(self.source)
    df = dataset['train'].to_pandas()
    # print(dataset)
    df = df[['answer', 'question','sentence']]
    grouped = df.groupby('sentence').agg({'answer': ' <sep> '.join, 'question': 'count'}).reset_index()
    grouped.rename(columns={'question': 'count'}, inplace=True)
    # save filtering for later
    # grouped = grouped[grouped['count'] >= int(self.config['aeMinAnswerCount'])]
    print(grouped.head())
    print(len(grouped))
    datasetDict = Dataset.from_pandas(grouped.reset_index(drop=True))
    print(dataset)
    for split, dataset in datasetDict.items():
      dataset.to_json(f"{self.destination}/{split}AE.jsonl")
    

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
  dp.process()


