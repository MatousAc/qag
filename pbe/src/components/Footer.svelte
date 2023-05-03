<script lang="ts">
	import Logo from "./Logo.svelte"
</script>

<div class="footer w-full flex items-center px-4 justify-end mt-auto">
  <!-- logo -->
  <a class="logo p-4" href="/testimony">
    <Logo class="w-16 md:w-20"/>
  </a>
</div>

<style lang="scss">
div.footer {
  bottom: 0;
  background-color: var(--nav);
  color: var(--nav-text);
  font-size: 1.2rem;
  border-top: var(--nav-border-bottom-width) var(--nav-border) solid;
}
</style>