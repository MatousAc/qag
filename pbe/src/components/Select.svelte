<script lang="ts">
import { onMount } from 'svelte'
import Row from '$comp/Row.svelte'

export let name: string
export let label: string
export let val = ''
export let options: { name: string; value: string }[] = []
export let ph = ''
let def: { name: string; value: string }

onMount(() => {
  const select = document.querySelector('select')
  if (ph) def = { name: ph, value: '' }
  else {
    def = options[0]
    options = options.slice(1)
  }
})
</script>

<Row justify="space-between">
  <label for={name}>
    {label}
  </label>
  <select {name} bind:value={val} on:change class="rounded-lg p-2">
    <option value="" disabled selected>{ph}</option>
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

select > option[disabled] {
  color: lightgray !important;
}
</style>
