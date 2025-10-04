// Bun build configuration for Scribe Frontend
import { defineConfig } from 'bun';

export default defineConfig({
  entrypoints: ['./src/index.js'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  
  // Build optimizations
  minify: process.env.NODE_ENV === 'production',
  sourcemap: true,
  splitting: true,
  
  // External dependencies that shouldn't be bundled
  external: [],
  
  // Plugin configuration
  plugins: [
    // Add custom plugins here if needed
  ],
  
  // Development configuration
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'process.env.DEBUG': JSON.stringify(process.env.DEBUG || 'false'),
  },
  
  // Asset handling
  loader: {
    '.css': 'css',
    '.svg': 'file',
    '.png': 'file',
    '.jpg': 'file',
    '.jpeg': 'file',
    '.gif': 'file',
    '.webp': 'file',
  },
  
  // Advanced build options
  naming: {
    entry: '[name].[hash].[ext]',
    chunk: '[name].[hash].[ext]',
    asset: '[name].[hash].[ext]',
  },
  
  // Bundle analysis
  metafile: true,
  
  // Performance optimizations
  treeShaking: true,
  
  // Error handling
  logLevel: 'info',
  
  // Watch mode configuration
  watch: process.env.NODE_ENV === 'development' ? {
    ignored: ['node_modules', 'dist', '*.log'],
  } : false,
});