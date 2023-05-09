export let nkjvData: {
  [book: string]: {
    [chapter: string]: {
      [verse: string]: string
    }
  }
}

export const loadNKJV = async () => {
  try {
    const response = await fetch(window.location + 'nkjv.json')
    nkjvData = await response.json()
    // Process the loaded data
  } catch (error) {
    console.log('Error fetching nkjv.json:', error)
  }
}

interface KeyVal {
  name: string
  value: string
}

// Get the list of books
export const getBooks = (): KeyVal[] => {
  return Object.keys(nkjvData).map(bookName => ({
    name: bookName,
    value: bookName
  }))
}

// get the number of chapters in a book
export const getChapters = (book: string): KeyVal[] => {
  const bookData = nkjvData[book]
  if (bookData) {
    return Object.keys(bookData).map(chapterName => ({
      name: chapterName,
      value: chapterName
    }))
  }
  return []
}

// get the number of verses in a chapter of a book
export const getVerses = (book: string, chapter: string): KeyVal[] => {
  const bookData = nkjvData[book]
  if (bookData) {
    const chapterData = bookData[chapter]
    if (chapterData) {
      return Object.keys(chapterData).map(verseName => ({
        name: verseName,
        value: verseName
      }))
    }
  }
  return []
}

export const getText = (
  Book: string,
  StartChapter: number,
  StartVerse: number,
  EndChapter: number | null = null,
  EndVerse: number | null = null
) => {
  // default to start positions
  EndChapter = EndChapter !== null ? EndChapter : StartChapter
  EndVerse = EndVerse !== null ? EndVerse : StartVerse

  let text = ''
  const BookData = nkjvData[Book]
  if (!BookData) return ''
  const ChapterData = BookData[StartChapter]
  if (!ChapterData) return ''

  for (let vrs in Object.keys(ChapterData)) {
    let verse = parseInt(vrs)
    if (verse >= StartVerse && verse <= EndVerse) {
      text += ChapterData[verse] + ' '
    }
  }
  return text
}
