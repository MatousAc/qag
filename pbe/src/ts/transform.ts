import { AutoTokenizer, T5ForConditionalGeneration } from 'web-transformers'

let tokenizer: AutoTokenizer
let model: T5ForConditionalGeneration

const generateProgress = async (
  outputTokenIds: number[],
  forInputIds: number[]
) => {
  let shouldContinue = true
  return shouldContinue
}
const generationOptions = {
  maxLength: 512,
  topK: 0
}

// load the tokenizer and model
export const loadModel = async (modelID: string, modelPath: string) => {
  tokenizer = AutoTokenizer.fromPretrained(modelID, modelPath)
  model = new T5ForConditionalGeneration(modelID, modelPath, async progress => {
    console.log(`Loading network: ${Math.round(progress * 100)}%`)
  })
  let gen: string = 'In the beginning, God created the heavens and the earth.'
  model.generate(
    await tokenizer.encode(gen),
    generationOptions,
    generateProgress
  )
}

export const generateQuestion = async () => {
  let verse: string = 'In the beginning, God created the heavens and the earth.'
  const inputTokenIds = await tokenizer.encode(verse)

  const finalOutputTokenIds = await model.generate(
    inputTokenIds,
    generationOptions,
    generateProgress
  )
  const finalOutput = (await tokenizer.decode(finalOutputTokenIds, true)).trim()
  console.log(finalOutput)
}
