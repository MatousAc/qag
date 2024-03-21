import re

class Verse():
  '''Represents a text from the Bible
  that can span multiple verses. Also
  includes context verses.'''
  # content stuff
  previous: str
  text: str
  following: str
  inContext: str
  questionContext: str
  wordCount: int

  # reference stuff
  ref: str
  book: str
  chapter: int
  start: int
  end: int
  refMatch = r'\s*(?P<book>(?:\d\s)?[a-zA-Z]+)\s(?P<chapter>\d+):(?P<start>\d+)(?:[-,]?[ ]?(?P<end>\d+))?,?\s*'
  
  def __init__(self, *args): 
    # 1 arg -> ref
    if len(args) == 1 and isinstance(args[0], str):
      self.ref = args[0]
      match = re.match(self.refMatch, self.ref)
      self.book = match.group('book')
      self.chapter = match.group('chapter')
      self.start = match.group('start')
      self.end = match.group('end') if match.group('end') else self.start
    # 3-4 args -> book, chapter, verses
    elif len(args) >= 3 and len(args) <= 4:
      self.book = args[0]
      self.chapter = args[1]
      self.start = args[2]
      self.end = args[3] if len(args) > 3 else self.start
    else: raise ValueError("improper number of arguments sent to Verse constructor")
    # convert strings to ints
    self.chapter = int(self.chapter)
    self.start = int(self.start)
    self.end = int(self.end)
    if self.end < self.start: raise ValueError('end is less than start')
    # format ref uniformly
    endStr = '-' + str(self.end) if self.end != self.start else ''
    self.ref = f'{self.book} {self.chapter}:{self.start}{endStr}'
