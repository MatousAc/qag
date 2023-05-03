<!-- <script lang="ts">
	import { onMount } from "svelte";

  let light = "/logoLight.png"
  let dark = "/logoDark.png"
  let isLightQuery = "(prefers-color-scheme: light)"
  let iconLink: HTMLLinkElement

  let systemTheme = (isLight: boolean) => {
    if (isLight) {
      iconLink.setAttribute("href", light)
    } else {
      iconLink.setAttribute("href", dark)
    }
  }
  
  onMount(() => {
    systemTheme(window.matchMedia(isLightQuery).matches)
    window.matchMedia(isLightQuery)
      .addEventListener('change', 
      ({ matches }) => systemTheme(matches))
  })
</script>

<svelte:head>
  <link bind:this={iconLink} rel="icon">
</svelte:head> -->

<svelte:head>
  <link rel="icon" href="/logoCathemeral.png">
</svelte:head>