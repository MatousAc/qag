<script lang='ts'>
import { onMount } from 'svelte'
import H2 from '$comp/H2.svelte'
import P from '$comp/P.svelte'
import Select from '$comp/Select.svelte'
import TextMedia from '$comp/TextMedia.svelte'
import Button from '$comp/Button.svelte'
import Row from '$/components/Row.svelte'
import { loadModel, generateQuestion } from '$/ts/model'
import { getBooks, getChapters, getVerses, getText } from '$/ts/nkjv'

let {book, chapter, startVerse, endVerse, text, result} = {
  book: 'Genesis', chapter: '1', startVerse: '1', endVerse: '1', text: '', result: ''
}
let chapters: { name: string; value: string }[] = getChapters(book)
let verses: { name: string; value: string }[] = getVerses(book, chapter)

const updateText = () => {
  text = getText(book, parseInt(chapter), parseInt(startVerse), 
    parseInt(chapter), parseInt(endVerse)
  )
}

onMount(async () => {
  loadModel('potsawee/t5-large-generation-squad-QuestionAnswer', 'https://storage.googleapis.com/aqg_onnx_models')
  text = getText("Genesis", 1, 1, 1, 1)
  console.log(text)
})

const truncateText = (text: string) => {
  const sentences = text.trim().split('. ');
  if (sentences.length <= 2) {
    return text;
  }

  const firstTwoSentences = sentences.slice(0, 2).join('. ');

  const words = text.trim().split(/\s+/);
  const lastEightWords = words.slice(-8).join(' ');

  const truncatedText = `${firstTwoSentences} ... ${lastEightWords}`;
  return truncatedText;
}

</script>

<svelte:head>
  <title>PBE Question Generator</title>
</svelte:head>

<TextMedia>
  <div slot='text'>
    <H2>Select a text to generate questions on.</H2>
    <Row justify='start'>
      <Select bind:value={book} name='book' options={getBooks()} on:change={() => {
        chapters = getChapters(book)
        chapter = '1'
        updateText()
      }} />
      <Select bind:value={chapter} name='chapter' options={chapters} on:change={() => {
        verses = getVerses(book, chapter)
        startVerse = '1'
        endVerse = '1'
        updateText()
      }}/>
      <H2>:</H2>
      <Select bind:value={startVerse} name='startVerse' options={verses} on:change={() => {
        if (parseInt(startVerse) > parseInt(endVerse)) {
          endVerse = startVerse
        }
        updateText()
      }}/>
      <H2>-</H2>
      <Select bind:value={endVerse} name='endVerse' options={verses} on:change={() => {
        if (parseInt(startVerse) > parseInt(endVerse)) {
          startVerse = endVerse
        }
        updateText()
      }}/>
    </Row>
    <P>
      {truncateText(text)}
    </P>
    <Row justify='between'>
      <Button onClick={() => generateQuestion(text)} class='mr-auto'>
        Generate Question
      </Button>
    </Row>
  </div>
  <textarea
    slot='media'
    class='p-4 mx-4 md:m-0 bg-transparent rounded-lg h-60'
  />
</TextMedia>

<style>
textarea {
  border: 2px solid var(--accent);
  resize: none;
  width: -webkit-fill-available;
}
</style>
