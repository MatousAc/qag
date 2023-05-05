<script lang="ts">
import { onMount } from 'svelte'
import H2 from '$comp/H2.svelte'
import Select from '$comp/Select.svelte'
import Input from '$comp/Input.svelte'
import TextMedia from '$comp/TextMedia.svelte'
import Button from '$comp/Button.svelte'
import Row from '$/components/Row.svelte'
import { AutoTokenizer, T5ForConditionalGeneration } from 'web-transformers'


let tokenizer: AutoTokenizer
let model: T5ForConditionalGeneration

const generateProgress = async (outputTokenIds: number[], forInputIds: number[]) => {
  let shouldContinue = true
  return shouldContinue
}
const generationOptions = {
  "maxLength": 512,
  "topK": 0
}

// load the tokenizer and model
const loadModel = async (modelID: string, modelPath:string) => {
  tokenizer = AutoTokenizer.fromPretrained(modelID, modelPath)
  model = new T5ForConditionalGeneration(modelID, modelPath, async progress => {
    console.log(`Loading network: ${Math.round(progress * 100)}%`)
  })
  let gen: string = "In the beginning, God created the heavens and the earth."
  model.generate(await tokenizer.encode(gen), generationOptions, generateProgress)
}

const generateQuestion = async () => {
  let verse: string = "In the beginning, God created the heavens and the earth."
  const inputTokenIds = await tokenizer.encode(verse)

  const finalOutputTokenIds = await model.generate(inputTokenIds, generationOptions, generateProgress)
  const finalOutput = (await tokenizer.decode(finalOutputTokenIds, true)).trim()
  console.log(finalOutput)
}

onMount(async () => {
  loadModel("potsawee/t5-large-generation-squad-QuestionAnswer", "/fastT5/")
})
</script>

<svelte:head>
  <title>PBE Question Generator</title>
</svelte:head>

<TextMedia>
  <div slot="text">
    <H2>Select a text to generate questions on.</H2>
    <Select name="book" label="Book" ph="Genesis" />
    <Input name="chapter" label="Chapter" ph="3" />
    <Input name="verse" label="Verse" ph="5" />
    <Row justify="between">
      <Button onClick={generateQuestion} class="mr-auto">
        Generate Question
      </Button>
    </Row>
  </div>
  <textarea
    slot="media"
    class="p-4 mx-4 md:m-0 bg-transparent rounded-lg h-60"
  />
</TextMedia>

<style>
textarea {
  border: 2px solid var(--accent);
  resize: none;
  width: -webkit-fill-available;
}
</style>
