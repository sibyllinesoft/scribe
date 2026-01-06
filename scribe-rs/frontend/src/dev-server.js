// Development server for React Arborist component testing
import { serve } from 'bun';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';

const PORT = process.env.PORT || 3000;
const isDev = process.env.NODE_ENV !== 'production';

// MIME types for different file extensions
const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.eot': 'application/vnd.ms-fontobject',
};

// Simple HTML template for development
const htmlTemplate = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scribe Frontend Development</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background-color: #f5f5f5;
        }
        
        .dev-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dev-header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .dev-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 400px;
        }
        
        .status {
            color: #28a745;
            font-weight: bold;
        }
        
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="dev-container">
        <div class="dev-header">
            <h1>Scribe Frontend Development Server</h1>
            <p class="status">✅ Development server running on port ${PORT}</p>
            <p>This server provides hot reloading for React Arborist component development.</p>
        </div>
        
        <div class="dev-content">
            <div id="root">
                <div class="loading">Loading React components...</div>
            </div>
        </div>
    </div>
    
    <!-- Load the compiled bundle -->
    <script type="module" src="/dist/index.js"></script>
    
    ${isDev ? `
    <!-- Development hot reload script -->
    <script>
        // Simple hot reload for development
        let lastModified = Date.now();
        
        function checkForUpdates() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.lastModified > lastModified) {
                        console.log('🔄 Reloading due to file changes...');
                        window.location.reload();
                    }
                    lastModified = data.lastModified;
                })
                .catch(error => {
                    console.warn('Hot reload check failed:', error);
                });
        }
        
        // Check for updates every 1 second in development
        setInterval(checkForUpdates, 1000);
        
        // Also reload on focus (in case file changed while away)
        window.addEventListener('focus', checkForUpdates);
        
        console.log('🚀 Development mode: Hot reload enabled');
    </script>
    ` : ''}
</body>
</html>
`;

// Track last modification time for hot reload
let lastModified = Date.now();

// Handle API status endpoint for hot reload
function handleStatusEndpoint() {
  return new Response(JSON.stringify({
    status: 'ok',
    lastModified,
    timestamp: Date.now(),
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

// Handle serving the main HTML page
function handleHtmlPage() {
  return new Response(htmlTemplate, {
    headers: { 'Content-Type': 'text/html' },
  });
}

// Serve a static file with appropriate MIME type and caching
function serveStaticFile(pathname, defaultMime, cacheControl) {
  const filePath = join(process.cwd(), pathname.slice(1));

  try {
    if (existsSync(filePath)) {
      const file = readFileSync(filePath);
      const ext = extname(filePath);
      const mimeType = mimeTypes[ext] || defaultMime;

      return new Response(file, {
        headers: {
          'Content-Type': mimeType,
          'Cache-Control': cacheControl,
        },
      });
    }
  } catch (error) {
    console.error('Error serving file:', error);
  }
  return null;
}

// Return a 404 Not Found response
function notFoundResponse() {
  return new Response('Not Found', {
    status: 404,
    headers: { 'Content-Type': 'text/plain' },
  });
}

const server = serve({
  port: PORT,

  async fetch(request) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    // Hot reload API endpoint
    if (pathname === '/api/status') {
      return handleStatusEndpoint();
    }

    // Update last modified time (simple file change detection)
    if (isDev) {
      lastModified = Date.now();
    }

    // Serve the main HTML page
    if (pathname === '/' || pathname === '/index.html') {
      return handleHtmlPage();
    }

    // Serve static files from dist directory
    if (pathname.startsWith('/dist/')) {
      const cacheControl = isDev ? 'no-cache' : 'public, max-age=31536000';
      const response = serveStaticFile(pathname, 'application/octet-stream', cacheControl);
      if (response) return response;
    }

    // Serve files from src directory in development
    if (isDev && pathname.startsWith('/src/')) {
      const response = serveStaticFile(pathname, 'text/plain', 'no-cache');
      if (response) return response;
    }

    // 404 for everything else
    return notFoundResponse();
  },
  
  error(error) {
    console.error('Server error:', error);
    return new Response('Internal Server Error', { 
      status: 500,
      headers: { 'Content-Type': 'text/plain' },
    });
  },
});

console.log(`🚀 Development server running at http://localhost:${PORT}`);
console.log(`📁 Serving from: ${process.cwd()}`);
console.log(`🔥 Hot reload: ${isDev ? 'enabled' : 'disabled'}`);

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down development server...');
  server.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down development server...');
  server.stop();
  process.exit(0);
});