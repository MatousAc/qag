<script>
import Fill from '$comp/Fill.svelte'

export let reverse = false
</script>

<div class="textMedia lg:flex justify-space-between mb-20">
  {#if !reverse}
    <div class="m-5 lg:w-2/5 lg:mr-12 lg:mt-5">
      <slot name="text" />
    </div>
    <div class="w-full">
      <slot name="media" />
    </div>
  {:else}
    <div class="w-full">
      <slot name="media" />
    </div>
    <div class="m-5 lg:w-2/5 lg:ml-12 lg:mt-5">
      <slot name="text" />
    </div>
  {/if}
</div>
