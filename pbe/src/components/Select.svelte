<script lang='ts'>
import { onMount } from 'svelte'
import Row from '$comp/Row.svelte'

export let name: string
export let label: string = ''
export let value = ''
export let justify: string = 'space-between'
export let options: { name: string; value: string }[] = []
export let selectFirst = true
let def: { name: string; value: string } = { name: '', value: '' }
let select: HTMLSelectElement

onMount(() => {
  if (selectFirst) {
    select.value = options[0]?.value
  }
})
</script>

<Row {justify} class='mr-4'>
  <label for={name} class='mr-2 {name ? '' : 'hidden'}'>
    {label}
  </label>
  <select {name} bind:this={select} bind:value={value} on:change class='rounded-lg p-2'>
    {#each options as { name, value }}
      <option {value}>{name}</option>
    {/each}
  </select>
</Row>

<style>
select {
  background-color: transparent;
  border: 2px solid var(--text);
  color: var(--text);
}

select > option {
  color: black;
}
</style>
