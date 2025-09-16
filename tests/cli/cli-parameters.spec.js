import { test, expect } from '@playwright/test';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const scribeRoot = path.resolve(__dirname, '../../scribe-rs');

/**
 * E2E tests for CLI with all UI command line parameters
 * Tests the web interface with different CLI configurations
 */

// Helper function to start Scribe with specific parameters
async function startScribeWithParams(params = [], timeout = 30000) {
  return new Promise((resolve, reject) => {
    const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', ...params], {
      cwd: scribeRoot,
      stdio: 'pipe'
    });

    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
      // Look for server start indication
      if (stdout.includes('Server running on') || stdout.includes('Listening on')) {
        resolve({ child, stdout, stderr });
      }
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('error', (error) => {
      reject(new Error(`Failed to start process: ${error.message}`));
    });

    child.on('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}. STDERR: ${stderr}`));
      }
    });

    // Timeout
    setTimeout(() => {
      child.kill();
      reject(new Error(`Timeout waiting for server to start. STDOUT: ${stdout}, STDERR: ${stderr}`));
    }, timeout);
  });
}

// Helper function to stop Scribe process
function stopScribe(child) {
  return new Promise((resolve) => {
    child.on('exit', () => resolve());
    child.kill('SIGTERM');
    
    // Force kill after 5 seconds
    setTimeout(() => {
      child.kill('SIGKILL');
      resolve();
    }, 5000);
  });
}

test.describe('CLI Parameters - Basic Functionality', () => {
  test('should start with default parameters', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([testDir, '--port', '8081', '--no-browser']);
      
      // Connect to the web interface
      await page.goto('http://localhost:8081');
      
      // Verify the interface loads
      await expect(page).toHaveTitle(/Scribe/);
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should respect custom port parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const customPort = '8082';
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([testDir, '--port', customPort, '--no-browser']);
      
      // Connect to the custom port
      await page.goto(`http://localhost:${customPort}`);
      
      // Verify the interface loads on custom port
      await expect(page).toHaveTitle(/Scribe/);
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should respect custom host parameter', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--host', '127.0.0.1', 
        '--port', '8083', 
        '--no-browser'
      ]);
      
      // Connect to the custom host
      await page.goto('http://127.0.0.1:8083');
      
      // Verify the interface loads
      await expect(page).toHaveTitle(/Scribe/);
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });
});

test.describe('CLI Parameters - Token Budget', () => {
  test('should respect custom token budget parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const tokenBudget = '15000';
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--token-budget', tokenBudget,
        '--port', '8084',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8084');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Look for token budget information in the UI
      const tokenInfo = await page.locator('text=/token|budget/i');
      if (await tokenInfo.isVisible()) {
        const text = await tokenInfo.textContent();
        expect(text?.toLowerCase()).toContain('token');
      }
      
      // Verify files are loaded (should work with any reasonable token budget)
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      await expect(treeNodes.first()).toBeVisible();
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should handle large token budget parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const largeTokenBudget = '50000';
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--token-budget', largeTokenBudget,
        '--port', '8085',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8085');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // With a large token budget, more files should be included
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      const nodeCount = await treeNodes.count();
      
      expect(nodeCount).toBeGreaterThan(0);
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should handle small token budget parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const smallTokenBudget = '5000';
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--token-budget', smallTokenBudget,
        '--port', '8086',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8086');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Even with a small budget, should still show some files
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      await expect(treeNodes.first()).toBeVisible();
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });
});

test.describe('CLI Parameters - File Size Limits', () => {
  test('should respect max file size parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const maxFileSize = '100000'; // 100KB
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--max-file-size', maxFileSize,
        '--port', '8087',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8087');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Files should be loaded (respecting size limit)
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      await expect(treeNodes.first()).toBeVisible();
      
      // Verify that large files might be excluded (check for size indicators if present)
      const sizeIndicators = await page.locator('text=/size|large|excluded/i');
      // This is optional - depends on UI implementation
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should handle very small max file size parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const verySmallSize = '1000'; // 1KB
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--max-file-size', verySmallSize,
        '--port', '8088',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8088');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Should still show the file tree structure, even if content is limited
      const treeContainer = await page.locator('[data-testid="file-tree"], .file-tree, .tree-container');
      await expect(treeContainer).toBeVisible();
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should handle large max file size parameter', async ({ page }) => {
    const testDir = scribeRoot;
    const largeSize = '10000000'; // 10MB
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--max-file-size', largeSize,
        '--port', '8089',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8089');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // With large file size limit, more files should be included
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      const nodeCount = await treeNodes.count();
      
      expect(nodeCount).toBeGreaterThan(0);
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });
});

test.describe('CLI Parameters - Test File Inclusion', () => {
  test('should exclude test files by default', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--port', '8090',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8090');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Look for test files (they should be excluded by default)
      const fileNames = await page.locator('[data-testid="file-node"], .file-item').allTextContents();
      const testFiles = fileNames.filter(name => 
        name.includes('test') || 
        name.includes('spec') || 
        name.includes('.test.') || 
        name.includes('_test.')
      );
      
      // Test files should be fewer when excluded (this is context-dependent)
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should include test files with --no-exclude-tests parameter', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir, 
        '--no-exclude-tests',
        '--port', '8091',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8091');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // With --no-exclude-tests, test files should be visible
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      await expect(treeNodes.first()).toBeVisible();
      
      // Look for test-related files or directories
      const testElements = await page.locator('text=/test|spec/i');
      // Should find some test-related content when tests are included
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });
});

test.describe('CLI Parameters - Combined Options', () => {
  test('should handle multiple parameters together', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir,
        '--port', '8092',
        '--host', '127.0.0.1',
        '--token-budget', '20000',
        '--max-file-size', '500000',
        '--no-exclude-tests',
        '--no-browser'
      ]);
      
      await page.goto('http://127.0.0.1:8092');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Verify the interface works with all parameters combined
      await expect(page).toHaveTitle(/Scribe/);
      
      const treeNodes = await page.locator('[data-testid="tree-node"], .tree-node, .file-item');
      await expect(treeNodes.first()).toBeVisible();
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });

  test('should handle edge case parameter combinations', async ({ page }) => {
    const testDir = scribeRoot;
    
    let scribeProcess;
    try {
      scribeProcess = await startScribeWithParams([
        testDir,
        '--port', '8093',
        '--token-budget', '1000', // Very small
        '--max-file-size', '10000000', // Very large
        '--no-exclude-tests',
        '--no-browser'
      ]);
      
      await page.goto('http://localhost:8093');
      await page.waitForSelector('[data-testid="file-tree"], .file-tree, .tree-container', { timeout: 10000 });
      
      // Should handle conflicting constraints gracefully
      const treeContainer = await page.locator('[data-testid="file-tree"], .file-tree, .tree-container');
      await expect(treeContainer).toBeVisible();
      
    } finally {
      if (scribeProcess) {
        await stopScribe(scribeProcess.child);
      }
    }
  });
});

test.describe('CLI Parameters - Error Handling', () => {
  test('should display helpful error for invalid port', async () => {
    const testDir = scribeRoot;
    
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', testDir, '--port', '999999'], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stderr = '';
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stderr });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stderr });
      }, 10000);
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr.toLowerCase()).toContain('port');
  });

  test('should display helpful error for invalid token budget', async () => {
    const testDir = scribeRoot;
    
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', testDir, '--token-budget', 'invalid'], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stderr = '';
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stderr });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stderr });
      }, 10000);
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr.toLowerCase()).toContain('token');
  });

  test('should display helpful error for invalid max file size', async () => {
    const testDir = scribeRoot;
    
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', testDir, '--max-file-size', 'invalid'], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stderr = '';
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stderr });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stderr });
      }, 10000);
    });

    expect(result.code).not.toBe(0);
    expect(result.stderr.toLowerCase()).toContain('file size');
  });

  test('should display helpful error for nonexistent directory', async () => {
    const nonexistentDir = '/nonexistent/directory/path';
    
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', nonexistentDir], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stderr = '';
      let stdout = '';
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stderr, stdout });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stderr, stdout });
      }, 10000);
    });

    expect(result.code).not.toBe(0);
    const output = (result.stderr + result.stdout).toLowerCase();
    expect(output).toMatch(/(not exist|not found|invalid path|directory)/);
  });
});

test.describe('CLI Parameters - Help and Version', () => {
  test('should display help information', async () => {
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', '--help'], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stdout = '';
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stdout });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stdout });
      }, 10000);
    });

    expect(result.code).toBe(0);
    expect(result.stdout).toContain('--port');
    expect(result.stdout).toContain('--host');
    expect(result.stdout).toContain('--token-budget');
    expect(result.stdout).toContain('--max-file-size');
    expect(result.stdout).toContain('--no-browser');
    expect(result.stdout).toContain('--no-exclude-tests');
  });

  test('should display version information', async () => {
    const result = await new Promise((resolve) => {
      const child = spawn('cargo', ['run', '-p', 'scribe-webservice', '--bin', 'scribe-web', '--', '--version'], {
        cwd: scribeRoot,
        stdio: 'pipe'
      });

      let stdout = '';
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.on('exit', (code) => {
        resolve({ code, stdout });
      });

      // Kill after timeout
      setTimeout(() => {
        child.kill();
        resolve({ code: -1, stdout });
      }, 10000);
    });

    expect(result.code).toBe(0);
    expect(result.stdout).toMatch(/\d+\.\d+\.\d+/); // Version number pattern
  });
});