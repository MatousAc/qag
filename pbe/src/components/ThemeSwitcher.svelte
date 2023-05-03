<script lang="ts">
  import type { SiteTheme } from "../ts/types";
  import { theme } from "../ts/stores";
  import Fa from 'svelte-fa'
  import { faMoon, faSun, faDisplay, type IconDefinition } from '@fortawesome/free-solid-svg-icons'

  let selectedTheme: SiteTheme = "system";
  let themeIcon: IconDefinition;
  theme.subscribe(val => {
    selectedTheme = val
    if (selectedTheme === "system") {
      themeIcon = faDisplay
    } else if (selectedTheme === "light") {
      themeIcon = faSun
    } else {
      themeIcon = faMoon
    }
  })

  $: changeTheme = () => {
    if (selectedTheme === "system") {
      $theme = "light"
    } else if (selectedTheme === "light") {
      $theme = "dark"
    } else {
      $theme = "system"
    }
  };
</script>

<button
  class="do-transition md:py-1 px-3 rounded-3xl justify-center items-center {$$props.class}"
  on:click={changeTheme}
>
  <Fa class="icon" icon={themeIcon}/>
  <span class="label ml-2 md:ml-1">{selectedTheme}</span>
</button>

<style lang="scss">
button {
  background-color: var(--theme-background);
  border: 2px solid var(--theme-text);
  color: var(--theme-text);
  font-size: 0.9em;

  span {
    text-transform: capitalize;
  }
}

@media only screen and (max-width: 767px) {
  span {
    min-width: 6ch;
  }

  .icon {
    min-width: 2ch;
  }
}

@media only screen and (min-width: 768px) {
  span {
    min-width: 12ch;
  }

  span.label::after {
    content: " theme";
  }

  .icon {
    min-width: 2ch;
  }
}
</style>
