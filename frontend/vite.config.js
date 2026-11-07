import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const applicationEntry = fileURLToPath(new URL('./index.html', import.meta.url));

export default defineConfig({
  // Relative assets keep the static bundle hostable at any path (repo subpath
  // or domain root); the app has no router and no absolute fetches.
  base: './',
  server: {
    fs: {
      allow: [repositoryRoot],
    },
  },
  build: {
    rollupOptions: {
      input: applicationEntry,
    },
  },
});
