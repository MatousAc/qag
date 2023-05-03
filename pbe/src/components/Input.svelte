<script lang="ts">
import Row from '$comp/Row.svelte'

export let name: string
export let label: string
export let ph = ''
export let val = ''
export let autofocus = false
</script>

<Row align="flex-end" justify="between">
  <label for={name} class="mr-8">
    {label}
  </label>
  <!-- svelte-ignore a11y-autofocus -->
  <input
    {name}
    bind:value={val}
    on:change
    placeholder={ph}
    {autofocus}
    class="rounded-lg p-2"
  />
</Row>

<style>
input {
  border: 2px solid var(--text);
  background-color: transparent;
  color: var(--text);
  max-width: 50%;
}
</style>
