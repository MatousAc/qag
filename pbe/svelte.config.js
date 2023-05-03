import adapter from '@sveltejs/adapter-auto'
import { vitePreprocess } from '@sveltejs/kit/vite'

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://kit.svelte.dev/docs/integrations#preprocessors
  // for more information about preprocessors
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter(),
    files: {
      assets: 'src/assets',
      routes: 'src/routes'
    },
    alias: {
      // this will match a directory and its contents
      // (`my-directory/x` resolves to `path/to/my-directory/x`)
      $: './src',
      $comp: './src/components',
      $ts: './src/ts'
    }
  }
}

export default config
