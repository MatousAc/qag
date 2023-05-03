<script>
export let justify = 'center'
export let align = 'center'
</script>

<div class="flex flex-col justify-{justify} align-{align} {$$props.class}">
  <slot />
</div>
