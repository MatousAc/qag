<script>
export let onClick = () => {}
export let outlined = true
</script>

<button
  class="font-bold py-3 px-5 my-6 rounded {$$props.class}{outlined
    ? ' outlined'
    : ''}"
  on:click={onClick}
  style={$$props.style}
>
  <slot />
</button>

<style>
button {
  border: none;
  cursor: pointer;
  color: var(--nav-underline);
  background-color: var(--nav);
}

button.outlined {
  border: 2px solid var(--nav-border);
}
</style>
