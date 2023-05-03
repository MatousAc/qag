<script lang="ts">
	import { onMount } from "svelte";
  import { theme } from "../ts/stores";

  let light = "#264863"
  let dark = "#17121f"
  let isLightQuery = "(prefers-color-scheme: light)"
  let meta: HTMLMetaElement

  let systemTheme = (isLight: boolean) => {
    if (isLight) {
      meta.setAttribute("content", light)
    } else {
      meta.setAttribute("content", dark)
    }
  }
  
  onMount(() => {
    theme.subscribe(val => {
      if (val === "system") {
        systemTheme(window.matchMedia(isLightQuery).matches)
        window.matchMedia(isLightQuery)
          .addEventListener('change', 
          ({ matches }) => systemTheme(matches))
      } else {
        systemTheme(val === "light")
      }
    })
  })
</script>

<svelte:head>
  <meta bind:this={meta} name="theme-color">
</svelte:head>
