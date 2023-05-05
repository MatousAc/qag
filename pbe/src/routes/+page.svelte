<script lang="ts">
import { onMount } from 'svelte'
import H2 from '$comp/H2.svelte'
import Select from '$comp/Select.svelte'
import Input from '$comp/Input.svelte'
import TextMedia from '$comp/TextMedia.svelte'
import Button from '$comp/Button.svelte'
import Row from '$/components/Row.svelte'
import { loadModel, generateQuestion } from '$ts/transform'

onMount(async () => {
  loadModel("potsawee/t5-large-generation-squad-QuestionAnswer", "https://storage.googleapis.com/aqg_onnx_models")
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
