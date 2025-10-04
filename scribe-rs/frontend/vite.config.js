import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      babel: {
        presets: ['@babel/preset-react']
      }
    })
  ],
  build: {
    lib: {
      entry: 'src/index.js',
      name: 'ScribeTreeBundle',
      fileName: 'scribe-tree-bundle',
      formats: ['umd']
    },
    rollupOptions: {
      output: {
        dir: '../templates/assets',
        globals: {
          'react': 'React',
          'react-dom': 'ReactDOM',
          'react-arborist': 'ReactArborist',
          'lucide-react': 'LucideReact'
        }
      }
    },
    minify: 'terser',
    sourcemap: false
  },
  define: {
    'process.env.NODE_ENV': '"production"'
  }
})