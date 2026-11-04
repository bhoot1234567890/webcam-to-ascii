import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const applicationEntry = fileURLToPath(new URL('./index.html', import.meta.url));

export default defineConfig({
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
