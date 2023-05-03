<script lang="ts">
import '$/app.postcss'
import '$/css/colors.css'
import Bar from '$comp/Bar.svelte'
import type { SiteTheme } from '$ts/types'
import { theme } from '$ts/stores'
import BrowserTheme from '$comp/BrowserTheme.svelte'
import Fonts from '$comp/Fonts.svelte'
import Favicon from '$comp/Favicon.svelte'
import Footer from '$comp/Footer.svelte'

let selectedTheme: SiteTheme
theme.subscribe(val => (selectedTheme = val))
</script>

<Fonts />
<Favicon />
<BrowserTheme />

<main class="skeleton flex flex-col {selectedTheme}">
  <Bar />
  <div class="w-full md:w-4/5 text-xl mx-auto md:px-8 my-10 md:my-20 max-w-7xl">
    <slot />
  </div>
  <Footer />
</main>

<style>
.skeleton {
  color-scheme: light dark;
  background-color: var(--background);
  color: var(--text);

  min-height: 100vh;
}
</style>
