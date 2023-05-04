<script lang="ts">
import { onMount } from 'svelte'
import H2 from '$comp/H2.svelte'
import Select from '$comp/Select.svelte'
import Input from '$comp/Input.svelte'
import TextMedia from '$comp/TextMedia.svelte'
import Button from '$comp/Button.svelte'
import Row from '$/components/Row.svelte'
import * as ort from 'onnxruntime-web'
import jsTokens from "js-tokens"

let tokenMap: Record<string, number> = {}
onMount(() => {
  fetch('/OptimumONNX/tokenMap.json')
  .then(response => response.json())
  .then(tm => tokenMap = tm)
  .catch(error => console.error(error));
})

let getFeed = (text: string) => {
  let inputArr: Float32Array = new Float32Array(512)
  let attentionArr: Float32Array = new Float32Array(512)
  
  let i = 0;
  for (const token of jsTokens(text)) {
    // ignore some tokens
    switch(token.type) {
      case "WhiteSpace": continue;
    }

    i++
    if (token.value in tokenMap) {
      inputArr[i] = tokenMap[token.value]
      attentionArr[i] = 1 // mark as valuable
    } else {
      inputArr[i] = tokenMap["<unk>"]
    }
  }

  const inputTensor = new ort.Tensor('float32', inputArr)
  const attentionTensor = new ort.Tensor('float32', attentionArr)
  
  return { input_ids: inputTensor, attention_mask: attentionTensor }
}

let generateQuestion = async () => {
  let verse = "In the beginning, God created the heavens and the earth."
  let feed = getFeed(verse)
  console.log(feed)

  // create a new session and load the specific model.
  const session = await ort.InferenceSession.create('/OptimumONNX/encoder_model.onnx')
  const results = await session.run(feed)

  // read from results
  const dataC = results.c.data
  console.log(`data of result tensor 'c': ${dataC}`)
}
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
