import pandas as pd, sys, os, random, csv, json, string, re
from datasets import load_dataset, Dataset
from configBase import ConfigBase
from verse import Verse

class DataProcessor(ConfigBase):
  '''Handles answer deduplication, some data cleaning,
  data reformating, data reporting, and getting Bible verse texts.'''
  def configure(self):
    self.source = self.paths['dpSource']
    self.destination = self.paths['dpDest']
    self.bibleSrc = self.basePath + '/data/bible/nkjv.csv'
    
    self.nkjv = pd.read_csv(self.bibleSrc)
    self.nkjvInfo = self.getNkjvInfo()
    fp = os.path.normpath(self.basePath + '/src/commonWords.txt')
    self.commonWords = open(fp).read().split()

  def getNkjvInfo(self):
    '''Creates an object representing the NKJV Bible. Structure:
    {Book: {numChapters: #chap, 1:#vs, 2#vs}, Book2:...}'''
    nkjvContent = {}

    with open(self.bibleSrc, 'r', encoding = 'utf-8') as csvFile:
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

  def enumerateContext(self, contexts):
    '''Takes the current generation contexts and
    enumerates them into all the verse references
    that need QA generation. Each list of verses
    indicates a separate output file. Saves results
    in context member variable.'''
    vsLists = []
    extendedVsLists = []
    # if we have a range of things, we split it up first
    if len(contexts) == 1 and '-' in contexts[0]:
        context = contexts[0]
        book, range_part = context.rsplit(' ', 1)
        start, end = map(int, range_part.split('-'))
        contexts = ', '.join(f'{book} {i}' for i in range(start, end + 1))
    # first we get all the verse references individually
    for text in contexts:
      verse = Verse(text)
      def appCh(ch):
        bkCh = f'{verse.book} {ch}'
        tempList = []
        for vs in range(1, self.nkjvInfo[verse.book][ch] + 1):
          tempList.append(f'{bkCh}:{vs}')
        vsLists.append(tempList)
      
      if verse.start: vsLists.append([verse.ref])
      elif verse.chapter: appCh(verse.chapter)
      else:
        for chapter in range(1, self.nkjvInfo[verse.book]['numChapters'] + 1):
          appCh(chapter)
    
    # next we insert bridge verses
    numRefs = len([vs for vsList in vsLists for vs in vsList])
    i = 0
    for vsList in vsLists: # for every list
      extendedVsList = [] # make empty inner list of verses
      vsInSent = 0 # number of verses in current sentence
      for ref in vsList: # go through each reference
        i += 1
        self.printProgressBar(i, numRefs, label = f'enumerating context')
        vsInSent += 1 # and count verses in current sentence
        # if the verse ends a sentence
        verse = self.constructVerse(ref)
        match = re.match(r'.*[\.!?:;][\'")]*$', verse.text)
        if match:
          if vsInSent == 2: # reset the count and add bridge if just 2 vrs
            extendedVsList.append(f'{verse.book} {verse.chapter}:{verse.start - vsInSent + 1}-{verse.start}')
          vsInSent = 0
        # end by adding current (last) verse
        extendedVsList.append(ref)
      extendedVsLists.append(extendedVsList) # collect all lists
    return extendedVsLists

  def smartUnCapitalize(self, str):
    if str.split()[0].lower() in self.commonWords:
      str = str[0].lower() + str[1:]
    return str

  def contextualizationLoop(self, df):
    '''Adds the actual verse text columns to Dataframe df'''
    maximum = len(df.index)
    for i, row in df.iterrows():
      # show progress
      self.printProgressBar(i, maximum = maximum, label = 'contextualizing')
      # get verse pieces
      book = row['book']; chapter = row['chapter']; start = row['verse']; end = row['endVerse']
      try: verse: Verse = self.constructVerse(book, chapter, start, end)
      except KeyboardInterrupt: self.printReplace('Closing')
      except: print(f'Error fetching verse {book} {chapter}:{verse}-{end} -> {row}')
      # assign pieces
      df.at[i, 'ref'] = verse.ref
      df.at[i, 'question'] = verse.ref + ', ' + df.at[i, 'question']
      df.at[i, 'sentence'] = verse.text
      df.at[i, 'paragraph'] = verse.inContext
      df.at[i, 'paragraph_sentence'] = f'{verse.previous}<hl> {verse.text} <hl>{verse.following}'
    self.printProgressBar(maximum, maximum = maximum, label = 'contextualizing') # done
    # drop rows that we can't get references for
    df = df.dropna(subset=['sentence', 'paragraph', 'paragraph_sentence'])
    print('\n')
    return df

  def pbeContextualize(self):
    '''Adds verse texts to data. Produces Ushio-like csv for AE=>QG training.'''
    df = pd.read_csv(self.source)
    if not self.quiet: print(f'dataset length: {len(df.index)}')
    df = self.contextualizationLoop(df)

    # reorganize columns
    cols = ['answer', 'question', 'sentence', 'paragraph', 'paragraph_sentence', 'points', 'source', 'quality', 'ref']
    df = df[cols]
    if not self.quiet: print(df.head())
    # drop rows that have sentences w/ more than 150 words
    df = df[df["sentence"].apply(lambda x: len(x.split()) <= 150)]
    # save
    df.to_csv(self.destination, index=False)

  def e2eAggregate(self):
    '''Aggregates Q&A by reference. Deduplicates QA based on answer similarity'''
    if '.csv' in self.source: df = pd.read_csv(self.source)
    else: df = load_dataset(self.source)['train'].to_pandas()
    
    # df = df[df['quality'] > 7]
    df['qa'] = 'Q: According to ' + df['question'] + ' A: ' + df['answer']
    df['count'] = 1
    
    # leverage existing AE deduplication to dedup Q&As
    def qaDeduplicate(qas):
      answers = [qa.split('A: ')[1] for qa in qas]
      dedupAnswers = self.aeDeduplicate(answers)
      dedupQAs = []
      for qa in qas:
        answer = qa.split('A: ')[1]
        if answer in dedupAnswers:
          dedupQAs.append(qa)
          dedupAnswers = list(filter(lambda a: a != answer, dedupAnswers))
      return dedupQAs
    
    grouped = df.groupby(['sentence', 'paragraph', 'ref']).agg({
      'qa': lambda x: self.sep.join(qaDeduplicate(x)),
      'points': 'mean',
      'quality': 'mean'
    }).reset_index()
    
    dataset = Dataset.from_pandas(grouped)
    dataset.to_json(self.destination)

  def aeFilter(self, answers, testing = False) -> list:
    '''Filters out badly generated answers'''
    # 1. remove answers longer than 25 words
    for answer in answers.copy():
      if len(answer.split()) > 25:
        answers.remove(answer)
        if testing: print('Removing:\n', answer, "\nbecause it is over 25 words.")

    # 2. remove multi-point answers with more than twice
    # as many words in any answer as there are total points
    pointRe = r'\s*\((\d+)\)\s*'
    for answer in answers.copy():
      matches = re.findall(pointRe, answer)
      totalPoints = max(len(matches), 1)
      if totalPoints == 1:
        # remove single-point answers that are too long
        if len(answer.split()) > 13:
          answers.remove(answer)
          if testing: print('Removing single point answer over 13 words:\n', answer)
        continue
          
      parts = re.split(pointRe, answer)
      rmList = []
      for part in parts:
        if len(part) < 3: rmList.append(part)
      for part in rmList: parts.remove(part)
      maxLen = 0
      # 3. rm if any parts are identical (ignore case)
      currAnsLen = len(answers)
      for part in parts:
        partCopy = parts.copy()
        partCopy.remove(part)
        if part.lower() in [p.lower() for p in partCopy]:
          answers.remove(answer)
          if testing: print('Removing:\n', answer, "\ndue to identical parts.")
          break
      if currAnsLen != len(answers): continue # removed this answer, move on
      # 4. remove if there are too many words in one of the points
      for part in parts:
        l = len(part.split())
        if maxLen < l: maxLen = l
      if maxLen > (totalPoints + 5): # slightly scale with point values
        if testing: print('Removing:\n', answer, "\ndue to a bad balance of words to points.")
        answers.remove(answer)
        continue
      # 5. if 2-point answer parts differ only by first word, rm
      if len(parts) == 2:
        pt0 = parts[0].lower().strip().strip(' and'); pt1 = parts[1].lower().strip()
        l0 = len(pt0.split()); l1 = len(pt1.split())
        if l0 > l1: pt0 = ' '.join(pt0.split()[1:])
        else : pt1 = ' '.join(pt1.split()[1:])
        if pt1 == pt0:
          if testing: print('Removing:\n', answer, "\nbecause point parts only differ by first word.")
          answers.remove(answer)
    return answers

  def aeDeduplicate(self, answers, testing = False):
    '''Set-based deduplication of similar answers'''
    answers = list(answers)
    def normAns(text):
      '''Reduces text to basic words'''
      text = text.lower()
      text = re.sub(r'\((\d{1,3})\) ', '', text) # point marker removal
      # ignore punctuation
      text = text.translate(str.maketrans("", "", string.punctuation))
      return text
    
    def compare(text1, text2, threshold = 0.7):
      '''Determines whether two answers are similar, and if they
      are, which of the two to remove from answer list. returns -1 
      for keep left, 0 for keeping both, 1 for keeping right'''
      if text1 == text2: return -1 # identical
      # split texts into words
      words1 = set(normAns(text1).split())
      words2 = set(normAns(text2).split())
      # determine longer text, prefer keeping left (original) side
      longerText = -1 if len(text1.split()) >= len(text2.split()) else 1
      # if there is no 'and', base lengths off of word uniqueness
      len1 = len(words1); len2 = len(words2)
      if 'and' not in words1.union(words2): longerText = -1 if len1 >= len2 else 1
      if words1 == words2: return longerText # if set identical, keep original
      
      # answers are not similar if their length difference is
      # more than 50% of the smaller answer and more than 3 words
      maxLen = max(len1, len2); minLen = min(len1, len2)
      diff = abs(len1 - len2)
      if diff > max(int(minLen * 0.5), 3): return 0 # not similar
      
      # answers are similar if they intersect 'threshold' percent of
      # the time or only differ by one word for smaller answers
      if maxLen < 4:
        # different enough if each has a word the other doesn't have
        if len(words1.symmetric_difference(words2)) >= 2: return 0
        # ignore super common words that are in both answers
        fp = os.path.normpath(self.basePath + '/src/commonWords.txt')
        commonWords = open(fp).read().split()
        for word in commonWords:
          words1.discard(word)
          words2.discard(word)
        if words1 == words2: return longerText
        # maxNumSimilarWords = max(max(len(words1), len(words2)) - 2, 1)
      if maxLen < 6: maxNumSimilarWords = max(maxLen - 2, 1)
      else: maxNumSimilarWords = int(threshold * maxLen)
      if len(words1.intersection(words2)) <= maxNumSimilarWords: return 0 # not similar
      else: return longerText

    # collect the best unique answers
    uniqueElems = set()
    for currAns in answers:
      addFlag = True
      for uniqueElem in uniqueElems:
        decision = compare(uniqueElem, currAns)
        if decision == 0: continue # keep uniqueElem, compare current further
        if decision == 1: # replacement
          if testing: print(f'Replacing:\n' + uniqueElem + '\nwith:\n' + currAns)
          uniqueElems.remove(uniqueElem)
        elif decision == -1: # skip
          if testing: print(f'Not adding:\n' + currAns + '\nbecause it is close to:\n' + uniqueElem)
          addFlag = False
        break
      # only add if different from all
      if addFlag: uniqueElems.add(currAns)
    return list(uniqueElems)

  def makeAE(self):
    '''Aggregates verses and deduplicates
    answers to get AE data'''
    # load json of csv
    if '.csv' not in self.source:
      dataset = load_dataset(self.source)
      df = dataset['train'].to_pandas()
    else: df = pd.read_csv(self.source)
    df = df[['answer', 'question','sentence', 'quality']]
    qualityThreshold = int(self.cp['dataFormatter']['qualityThreshold'])
    df = df[df['quality'] > qualityThreshold]

    # group answers by verse and remove near duplicates
    grouped = df.groupby('sentence').agg({
      'answer': lambda x: self.sep.join(self.aeDeduplicate(x)),
      'quality': 'mean' 
    }).reset_index()
    grouped['count'] = grouped['answer'].apply(lambda x: x.count(sep) + 1)
    grouped.rename(columns={'question': 'count'}, inplace=True)
    # train model to produce 3-12 ans
    grouped = grouped[(grouped['count'] > 2)]
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
    
    v.text = ' '.join([getVrs(n) for n in range(v.start, v.end + 1)])
    v.inContext = f'{v.previous} {v.text} {v.following}'.strip()
    v.wordCount = len(v.text.split())
    v.questionContext = v.inContext if v.wordCount < 15 else v.text
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

  def modelExecTimes(self):
    '''Expects a CSV in dataSource config.'''
    data = pd.read_csv(self.source, sep=',')
    data['count'] = 1
    print(data)
    grouped = data.groupby(['mode', 'model']).agg({
      'count': 'sum',
      'elapsedTime': 'mean'
    }).reset_index()
    print(grouped)
    
  def dataReport(self):
    data = pd.read_csv(self.source, sep=',')
    data['count'] = 1
    total = len(data.index)
    print(f'{total} total unfiltered data rows')
    grouped = data.groupby(['source']).agg({
      'count': 'sum'
    }).reset_index()
    grouped['percentage'] = grouped['count'] / total
    print(grouped)

  def testAEFilter(self):
    ans = input('Enter answer list: ').lstrip('[\'').lstrip('["').rstrip('\']').rstrip('"]')
    # ans = re.split(r"', '|\", '|', \"|\", \"", ans)
    ans = re.split(r"[\"'], [\"']", ans)
    print('')
    ans = [a.strip("'") for a in ans]
    print(f'{len(ans)} answers @ start')
    print('\n'.join(ans) + '\n')
    ans = dp.aeFilter(ans, testing = True)
    ans = dp.aeDeduplicate(ans, testing = True)
    print('\n\nUnique Answers:')
    print('\n'.join(ans))
    print(f'{len(ans)} answers left')
  
  def qualityFilter(self):
    '''Processes a Dataset so that the quality 
    falls between two values. Saves to dest.'''
    dataset = load_dataset(
      'csv' if '.csv' in self.source else 'json',
      data_files = self.source
    )['train']
    print(dataset)
    print('Quality bounds are inclusive.')
    lowQ = input('Enter lower quality bound (enter for none): ')
    if lowQ == '': lowQ = 0
    else: lowQ = int(lowQ)
    highQ = input('Enter upper quality bound (enter for none): ')
    if highQ == '': highQ = 10
    else: highQ = int(highQ)
    # filter
    dataset = dataset.filter(lambda row: lowQ <= float(row['quality']) <= highQ)
    if '.csv' in self.destination: dataset.to_csv(self.destination)
    else: dataset.to_json(self.destination)

  def manualEvalStats(self):
    '''Computes statistics for manual evaluation.
    Expects folder to CSVs in dataSource config.'''
    csvFiles = [f.name for f in os.scandir(self.source) if f.is_file and '.csv' in f.name]
    df = pd.DataFrame()
    for file in csvFiles:
      data = pd.read_csv(os.path.normpath(self.source + '/' + file))
      df = pd.concat([df, data])
    totalRows = len(df.index)
    # general averages
    gram = df.loc[:, 'grammaticality'].mean()
    accept = df.loc[:, 'acceptability'].mean()
    print(f'Average grammaticality: {gram}')
    print(f'Average acceptability: {accept}')
    # percentage in each ranking for acceptablility
    df['count'] = 1
    groupedByRating = df.groupby('acceptability').agg({'count': 'sum'}).reset_index()
    groupedByRating['percentOfTotal'] = groupedByRating['count'] / totalRows
    print(groupedByRating[['acceptability', 'percentOfTotal']])
    # ratings based on contextType
    groupedByContextType = df.groupby('contextType').agg({
      'grammaticality': 'mean',
      'acceptability': 'mean'
    }).reset_index()
    print(groupedByContextType)
    print(f"Unique contexts: {df['reference'].nunique()}")


if __name__ == '__main__':
  dp = DataProcessor()
  match sys.argv[1].replace('-', '').lower():
    case 'randomverse': print(dp.getRandomVerse().text)
    case 'makeae': dp.makeAE()
    case 'testaefilter': dp.testAEFilter()
    case 'pbecontextualize': dp.pbeContextualize()
    case 'csvtojsonl': dp.csvToJsonl()
    case 'jsonltocsv': dp.jsonlToCsv()
    case 'modelexectimes': dp.modelExecTimes()
    case 'datareport': dp.dataReport()
    case 'qualityfilter': dp.qualityFilter()
    case 'e2eaggregate': dp.e2eAggregate()
    case 'manualevalstats': dp.manualEvalStats()
    case 'shownkjvinfo': print(dp.nkjvInfo)
    case 'none' | _: pass
