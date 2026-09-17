import {defineConfig} from 'vite';
export default defineConfig({
  server: {proxy: {
    '/api': 'http://127.0.0.1:8787',
    '/data': 'http://127.0.0.1:8787',
    '/stream': {target: 'ws://127.0.0.1:8787', ws: true},
  }},
  build: {
    target: 'es2022',
    rollupOptions: {input: {brain: 'index.html', race: 'race/index.html'}},
  },
});
