import { writable } from 'svelte/store'
import { AutoTokenizer, T5ForConditionalGeneration } from 'web-transformers'

let tokenizer: AutoTokenizer
let model: T5ForConditionalGeneration
export let loadProgress = writable('0%')

const generateProgress = async (
  outputTokenIds: number[],
  forInputIds: number[]
) => {
  let shouldContinue = true
  return shouldContinue
}
const generationOptions = {
  maxLength: 512,
  topK: 10
}

// load the tokenizer and model
export const loadModel = async (modelID: string, modelPath: string) => {
  tokenizer = AutoTokenizer.fromPretrained(modelID, modelPath)
  model = new T5ForConditionalGeneration(modelID, modelPath, async progress => {
    console.log(`Loading network: ${Math.round(progress * 100)}%`)
    loadProgress.set(`${Math.round(progress * 100)}%`)
  })
}

const processQA = (inputString: string) => {
  const re = /\[.*?\]/g
  let str = inputString.replace(re, '') // remove square brackets and their contents
  str = str.replace(/<.*?>/g, '')
  const parts = str.split('<s/>') // split string on '<s/>'
  if (parts.length < 2) {
    return ''
  }
  const question = parts[0].trim()
  const answer = parts[1].trim()

  return `Question: ${question}\nAnswer: ${answer}`
}

export const generateQuestion = async (text: string) => {
  const inputTokenIds = await tokenizer.encode(text)

  const finalOutputTokenIds = await model.generate(
    inputTokenIds,
    generationOptions,
    generateProgress
  )
  let finalOutput = (await tokenizer.decode(finalOutputTokenIds, false)).trim()
  finalOutput = processQA(finalOutput)
  console.log(finalOutput)
  return finalOutput
}
