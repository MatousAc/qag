<script lang="ts">
import H1 from '$comp/H1.svelte'
import H2 from '$comp/H2.svelte'
import Select from '$comp/Select.svelte'
import Input from '$comp/Input.svelte'
import TextMedia from '$comp/TextMedia.svelte'
import Button from '$comp/Button.svelte'
import Row from '$/components/Row.svelte'
import * as ort from 'onnxruntime-web'

async function generateQuestion() {
  console.log(ort)

  // create a new session and load the specific model.
  //
  // the model in this example contains a single MatMul node
  // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
  // it has 1 output: 'c'(float32, 3x3)
  const session = await ort.InferenceSession.create('/model.onnx')

  // prepare inputs. a tensor need its corresponding TypedArray as data
  const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  const dataB = Float32Array.from([
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
  ])
  const tensorA = new ort.Tensor('float32', dataA, [3, 4])
  const tensorB = new ort.Tensor('float32', dataB, [4, 3])

  // prepare feeds. use model input names as keys.
  const feeds = { a: tensorA, b: tensorB }

  // feed inputs and run
  const results = await session.run(feeds)

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
