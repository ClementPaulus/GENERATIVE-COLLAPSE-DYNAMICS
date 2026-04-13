import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
  site: 'https://calebpruett927.github.io',
  base: '/GENERATIVE-COLLAPSE-DYNAMICS/',
  integrations: [mdx(), tailwind()],
  // Domain routing: /finance/, /astronomy/, /standard_model/, etc.
  // Each domain gets its own hermetic section
  build: {
    format: 'directory',
  },
  vite: {
    // Allow importing JSON data files emitted by HCG
    json: {
      stringify: true,
    },
    build: {
      chunkSizeWarningLimit: 600, // Increase chunk size limit to 600 kB
    },
  },
});
