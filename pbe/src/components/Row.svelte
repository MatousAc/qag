<script>
export let justify = 'center'
export let align = 'center'
</script>

<div
  class="my-4 flex justify-{justify} align-{align} {$$props.class}"
  style={$$props.style}
>
  <slot />
</div>
