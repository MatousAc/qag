<script lang="ts">
import HeroIcon from '$comp/HeroIcon.svelte'
export let href: string
</script>

<div class="text-right">
  <a {href} class="flex justify-end items-center p-2 text-lg">
    <slot />
    <HeroIcon name="next" class="ml-2 text-4xl" />
  </a>
</div>
