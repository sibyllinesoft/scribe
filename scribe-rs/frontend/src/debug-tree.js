#!/usr/bin/env bun
// Standalone debugging script for React Arborist component integration
// Run with: bun run src/debug-tree.js

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  bold: '\x1b[1m',
  dim: '\x1b[2m'
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}✗${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.bold}${colors.cyan}=== ${msg} ===${colors.reset}`),
  subsection: (msg) => console.log(`\n${colors.bold}--- ${msg} ---${colors.reset}`),
  debug: (msg) => console.log(`${colors.dim}[DEBUG]${colors.reset} ${msg}`)
};

// Mock file data for testing
const mockFiles = [
  { path: 'src/index.js' },
  { path: 'src/components/TreeView.jsx' },
  { path: 'src/components/Button.jsx' },
  { path: 'src/components/Modal.jsx' },
  { path: 'src/utils/helpers.js' },
  { path: 'src/utils/api.js' },
  { path: 'src/hooks/useTreeData.js' },
  { path: 'src/styles/main.css' },
  { path: 'src/styles/components.css' },
  { path: 'tests/unit/component.test.js' },
  { path: 'tests/integration/app.test.js' },
  { path: 'tests/e2e/user-flow.test.js' },
  { path: 'docs/README.md' },
  { path: 'docs/API.md' },
  { path: 'package.json' },
  { path: '.gitignore' },
  { path: 'tsconfig.json' }
];

// Debug class for React Arborist troubleshooting
class ArboristDebugger {
  constructor() {
    this.issues = [];
    this.findings = [];
    this.recommendations = [];
  }

  addIssue(severity, category, message, details = null) {
    this.issues.push({ severity, category, message, details, timestamp: new Date().toISOString() });
  }

  addFinding(category, message, details = null) {
    this.findings.push({ category, message, details, timestamp: new Date().toISOString() });
  }

  addRecommendation(priority, message, action = null) {
    this.recommendations.push({ priority, message, action, timestamp: new Date().toISOString() });
  }

  // Test ScribeFileTree class instantiation
  async testScribeFileTreeCreation() {
    log.subsection('Testing ScribeFileTree Class Creation');
    
    try {
      // Mock DOM environment
      global.document = {
        getElementById: (id) => {
          log.debug(`Mock DOM: getElementById called with '${id}'`);
          return {
            scrollIntoView: () => log.debug('Mock scrollIntoView called'),
            innerHTML: '',
            appendChild: (child) => log.debug('Mock appendChild called'),
            style: {}
          };
        },
        createElement: (tag) => {
          log.debug(`Mock DOM: createElement called with '${tag}'`);
          return {
            className: '',
            style: {},
            appendChild: () => {},
            setAttribute: () => {},
            getAttribute: () => null,
            addEventListener: () => {},
            removeEventListener: () => {}
          };
        }
      };

      global.window = {
        React: { createElement: () => ({ type: 'mock' }) },
        ReactDOM: { createRoot: () => ({ render: () => log.debug('Mock React render called') }) },
        performance: { now: () => Date.now() }
      };

      global.console = {
        log: (...args) => log.debug(`Console.log: ${args.join(' ')}`),
        warn: (...args) => log.warning(`Console.warn: ${args.join(' ')}`),
        error: (...args) => log.error(`Console.error: ${args.join(' ')}`),
        groupCollapsed: () => {},
        groupEnd: () => {},
        trace: () => {},
        clear: () => {}
      };

      // Import and test ScribeFileTree
      const module = await import('./index.js');
      const ScribeFileTree = module.default;
      
      if (!ScribeFileTree) {
        this.addIssue('error', 'import', 'ScribeFileTree class not exported as default');
        return false;
      }

      const fileTree = new ScribeFileTree();
      
      if (!fileTree) {
        this.addIssue('error', 'instantiation', 'Failed to create ScribeFileTree instance');
        return false;
      }

      this.addFinding('instantiation', 'ScribeFileTree class created successfully');
      log.success('ScribeFileTree class instantiated successfully');
      return fileTree;
      
    } catch (error) {
      this.addIssue('error', 'instantiation', `ScribeFileTree creation failed: ${error.message}`, { stack: error.stack });
      log.error(`Failed to create ScribeFileTree: ${error.message}`);
      return false;
    }
  }

  // Validate a single tree node structure
  validateTreeNode(node, path = '') {
    if (!node.id) {
      this.addIssue('error', 'tree-structure', `Node missing id at path: ${path}`);
    }
    if (!node.name) {
      this.addIssue('error', 'tree-structure', `Node missing name at path: ${path}`);
    }
    if (typeof node.isFolder !== 'boolean') {
      this.addIssue('error', 'tree-structure', `Node isFolder not boolean at path: ${path}`);
    }
    if (node.isFolder && !Array.isArray(node.children)) {
      this.addIssue('warning', 'tree-structure', `Folder node missing children array at path: ${path}`);
    }

    if (node.children) {
      node.children.forEach((child, index) => {
        this.validateTreeNode(child, `${path}[${index}]`);
      });
    }
  }

  // Log tree structure recursively
  logTreeStructure(nodes, indent = '') {
    nodes.forEach(node => {
      log.debug(`${indent}${node.isFolder ? '📁' : '📄'} ${node.name} (${node.path})`);
      if (node.children) {
        this.logTreeStructure(node.children, indent + '  ');
      }
    });
  }

  // Test tree data building functionality
  testTreeDataBuilding(fileTree) {
    log.subsection('Testing Tree Data Building');

    if (!fileTree) {
      this.addIssue('error', 'tree-building', 'No fileTree instance available for testing');
      return false;
    }

    try {
      const treeData = fileTree.buildTreeData(mockFiles);

      if (!Array.isArray(treeData)) {
        this.addIssue('error', 'tree-building', 'buildTreeData did not return an array');
        return false;
      }

      if (treeData.length === 0) {
        this.addIssue('warning', 'tree-building', 'buildTreeData returned empty array');
        return false;
      }

      // Validate tree structure
      treeData.forEach((node, index) => {
        this.validateTreeNode(node, `root[${index}]`);
      });

      this.addFinding('tree-building', `Successfully built tree with ${treeData.length} root nodes`);
      log.success(`Tree data built successfully with ${treeData.length} root nodes`);

      log.debug('Tree structure:');
      this.logTreeStructure(treeData);

      return treeData;

    } catch (error) {
      this.addIssue('error', 'tree-building', `Tree building failed: ${error.message}`, { stack: error.stack });
      log.error(`Tree building failed: ${error.message}`);
      return false;
    }
  }

  // Build a node map from tree data for testing
  buildNodeMapFromTreeData(treeData) {
    const nodeMap = new Map();
    const addNodes = (nodes) => {
      nodes.forEach(node => {
        nodeMap.set(node.path, node);
        if (node.children) {
          addNodes(node.children);
        }
      });
    };
    addNodes(treeData);
    return nodeMap;
  }

  // Test file selection functionality
  testFileSelection(fileTree, nodeMap, testFilePath) {
    if (!nodeMap.has(testFilePath)) {
      this.addIssue('warning', 'checkbox', `Test file ${testFilePath} not found in tree`);
      return;
    }

    const initialState = fileTree.checkboxStates.get(testFilePath);
    if (!initialState || initialState.checked) return;

    fileTree.toggleFileCheckbox(nodeMap, testFilePath, true);
    const newState = fileTree.checkboxStates.get(testFilePath);

    if (newState.checked && fileTree.selectedFiles.has(testFilePath)) {
      this.addFinding('checkbox', `File selection works correctly for ${testFilePath}`);
      log.success('File checkbox toggling works correctly');
    } else {
      this.addIssue('error', 'checkbox', `File selection failed for ${testFilePath}`);
    }
  }

  // Test folder selection functionality
  testFolderSelection(fileTree, nodeMap, testFolderPath) {
    if (!nodeMap.has(testFolderPath)) return;

    fileTree.toggleFileCheckbox(nodeMap, testFolderPath, false);
    const folderState = fileTree.checkboxStates.get(testFolderPath);

    if (folderState.checked) {
      this.addFinding('checkbox', `Folder selection works correctly for ${testFolderPath}`);
      log.success('Folder checkbox toggling works correctly');
    } else {
      this.addIssue('error', 'checkbox', `Folder selection failed for ${testFolderPath}`);
    }
  }

  // Test checkbox state management
  testCheckboxManagement(fileTree, treeData) {
    log.subsection('Testing Checkbox State Management');

    if (!fileTree || !treeData) {
      this.addIssue('error', 'checkbox', 'Missing fileTree or treeData for checkbox testing');
      return false;
    }

    try {
      const nodeMap = this.buildNodeMapFromTreeData(treeData);

      // Test initial state
      if (fileTree.checkboxStates.size === 0) {
        this.addIssue('warning', 'checkbox', 'No checkbox states initialized');
      } else {
        this.addFinding('checkbox', `Initialized ${fileTree.checkboxStates.size} checkbox states`);
      }

      this.testFileSelection(fileTree, nodeMap, 'src/index.js');
      this.testFolderSelection(fileTree, nodeMap, 'src');

      return true;

    } catch (error) {
      this.addIssue('error', 'checkbox', `Checkbox testing failed: ${error.message}`, { stack: error.stack });
      log.error(`Checkbox testing failed: ${error.message}`);
      return false;
    }
  }

  // Test React component creation
  testComponentCreation(fileTree) {
    log.subsection('Testing React Component Creation');
    
    if (!fileTree) {
      this.addIssue('error', 'component', 'No fileTree instance for component testing');
      return false;
    }

    try {
      // Test node component creation
      const NodeComponent = fileTree.createNodeComponent();
      if (typeof NodeComponent !== 'function') {
        this.addIssue('error', 'component', 'createNodeComponent did not return a function');
      } else {
        this.addFinding('component', 'Node component created successfully');
      }

      // Test tree component creation
      const TreeComponent = fileTree.createTreeComponent();
      if (typeof TreeComponent !== 'function') {
        this.addIssue('error', 'component', 'createTreeComponent did not return a function');
      } else {
        this.addFinding('component', 'Tree component created successfully');
      }

      // Test component initialization
      fileTree.initializeTreeComponent();
      if (typeof fileTree.FileTreeComponent !== 'function') {
        this.addIssue('error', 'component', 'FileTreeComponent not properly initialized');
      } else {
        this.addFinding('component', 'FileTreeComponent initialized successfully');
        log.success('React components created and initialized successfully');
      }

      return true;
      
    } catch (error) {
      this.addIssue('error', 'component', `Component creation failed: ${error.message}`, { stack: error.stack });
      log.error(`Component creation failed: ${error.message}`);
      return false;
    }
  }

  // Test tree rendering simulation
  testTreeRendering(fileTree) {
    log.subsection('Testing Tree Rendering');
    
    if (!fileTree) {
      this.addIssue('error', 'rendering', 'No fileTree instance for rendering testing');
      return false;
    }

    try {
      // Test with missing container
      const result1 = fileTree.renderTree('non-existent-container', mockFiles);
      if (result1 !== false) {
        this.addIssue('warning', 'rendering', 'renderTree should return false for missing container');
      } else {
        this.addFinding('rendering', 'Correctly handles missing container');
      }

      // Test with mock container
      let renderCalled = false;
      global.window.ReactDOM.createRoot = () => ({
        render: () => { 
          renderCalled = true;
          log.debug('Mock render method called');
        }
      });

      const result2 = fileTree.renderTree('mock-container', mockFiles);
      if (result2 && renderCalled) {
        this.addFinding('rendering', 'Tree rendering simulation successful');
        log.success('Tree rendering test passed');
      } else {
        this.addIssue('warning', 'rendering', 'Tree rendering simulation issues detected');
      }

      return true;
      
    } catch (error) {
      this.addIssue('error', 'rendering', `Rendering test failed: ${error.message}`, { stack: error.stack });
      log.error(`Rendering test failed: ${error.message}`);
      return false;
    }
  }

  // Check a single global dependency
  checkGlobalDependency(name, required = true) {
    if (typeof window === 'undefined') return false;

    const isAvailable = !!window[name];
    if (isAvailable) {
      this.addFinding('template', `${name} is available globally`);
    } else if (required) {
      this.addIssue('error', 'template', `${name} not available globally`);
      this.addRecommendation('high', `Ensure ${name} is loaded before ScribeFileTree`, `Add ${name} script tag before bundle`);
    } else {
      this.addIssue('warning', 'template', `${name} not available globally yet (may be expected)`);
    }
    return isAvailable;
  }

  // Check for potential template integration issues
  checkTemplateIntegration() {
    log.subsection('Checking Template Integration Issues');

    try {
      if (typeof window === 'undefined') {
        this.addIssue('info', 'template', 'Running in Node.js environment (not browser)');
        return true;
      }

      this.addFinding('template', 'window object is available');
      this.checkGlobalDependency('React', true);
      this.checkGlobalDependency('ReactDOM', true);
      this.checkGlobalDependency('ScribeFileTree', false);

      if (typeof document !== 'undefined' && document.getElementById) {
        this.addFinding('template', 'document.getElementById available');
      } else if (typeof document !== 'undefined') {
        this.addIssue('error', 'template', 'document.getElementById not available');
      }

      return true;

    } catch (error) {
      this.addIssue('error', 'template', `Template integration check failed: ${error.message}`);
      return false;
    }
  }

  // Generate HTML test template for visual debugging
  generateTestTemplate() {
    log.subsection('Generating HTML Test Template');
    
    const template = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Arborist Debug Test</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        
        .debug-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .debug-header {
            background: #2196F3;
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .debug-content {
            padding: 20px;
        }
        
        .debug-section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }
        
        .debug-section h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        #tree-container {
            min-height: 400px;
            border: 2px dashed #ccc;
            border-radius: 6px;
            padding: 20px;
            background: #fafafa;
        }
        
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        
        .debug-buttons {
            margin: 20px 0;
        }
        
        .debug-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        
        .debug-btn:hover {
            background: #0056b3;
        }
        
        .debug-btn.danger {
            background: #dc3545;
        }
        
        .debug-btn.danger:hover {
            background: #c82333;
        }
        
        .log-output {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        .tree-controls {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="debug-container">
        <div class="debug-header">
            <h1>🌳 React Arborist Debug Test</h1>
            <p>Debugging ScribeFileTree component integration</p>
        </div>
        
        <div class="debug-content">
            <div class="debug-section">
                <h3>🔧 Debug Controls</h3>
                <div class="debug-buttons">
                    <button class="debug-btn" onclick="debugTests.runAllTests()">Run All Tests</button>
                    <button class="debug-btn" onclick="debugTests.testTreeCreation()">Test Tree Creation</button>
                    <button class="debug-btn" onclick="debugTests.testTreeBuilding()">Test Tree Building</button>
                    <button class="debug-btn" onclick="debugTests.testCheckboxes()">Test Checkboxes</button>
                    <button class="debug-btn" onclick="debugTests.renderTree()">Render Tree</button>
                    <button class="debug-btn danger" onclick="debugTests.clearLogs()">Clear Logs</button>
                </div>
                
                <div class="tree-controls">
                    <label for="file-count">Mock Files Count: </label>
                    <input type="range" id="file-count" min="5" max="50" value="17" onchange="debugTests.updateFileCount(this.value)">
                    <span id="file-count-display">17</span>
                    
                    <label for="max-depth" style="margin-left: 20px;">Max Depth: </label>
                    <input type="range" id="max-depth" min="2" max="8" value="4" onchange="debugTests.updateMaxDepth(this.value)">
                    <span id="max-depth-display">4</span>
                </div>
            </div>
            
            <div class="debug-section">
                <h3>📊 Test Status</h3>
                <div id="status-container">
                    <div class="status info">Ready to run tests...</div>
                </div>
            </div>
            
            <div class="debug-section">
                <h3>🌳 Tree Container</h3>
                <div id="tree-container">
                    Tree will render here...
                </div>
            </div>
            
            <div class="debug-section">
                <h3>📋 Debug Logs</h3>
                <div id="log-output" class="log-output">
[${new Date().toISOString()}] Debug environment ready...
                </div>
            </div>
            
            <div class="debug-section">
                <h3>ℹ️ System Information</h3>
                <div id="system-info">
                    <p><strong>User Agent:</strong> <span id="user-agent"></span></p>
                    <p><strong>React Available:</strong> <span id="react-status"></span></p>
                    <p><strong>React Arborist Available:</strong> <span id="arborist-status"></span></p>
                    <p><strong>ScribeFileTree Available:</strong> <span id="scribe-status"></span></p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Load dependencies -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script type="module">
        import { Tree } from 'https://unpkg.com/react-arborist@3.4.0/dist/react-arborist.esm.js';
        window.ReactArborist = { Tree };
        
        // Debug utilities
        class DebugTests {
            constructor() {
                this.fileTree = null;
                this.mockFileCount = 17;
                this.maxDepth = 4;
                this.updateSystemInfo();
            }
            
            log(message, type = 'info') {
                const timestamp = new Date().toISOString();
                const logOutput = document.getElementById('log-output');
                const colorCode = {
                    info: '\\x1b[36m',
                    success: '\\x1b[32m',
                    warning: '\\x1b[33m',
                    error: '\\x1b[31m'
                }[type] || '\\x1b[37m';
                
                logOutput.textContent += \`[\${timestamp}] \${colorCode}\${message}\\x1b[0m\\n\`;
                logOutput.scrollTop = logOutput.scrollHeight;
            }
            
            addStatus(message, type = 'info') {
                const container = document.getElementById('status-container');
                const statusDiv = document.createElement('div');
                statusDiv.className = \`status \${type}\`;
                statusDiv.textContent = message;
                container.appendChild(statusDiv);
            }
            
            clearLogs() {
                document.getElementById('log-output').textContent = \`[\${new Date().toISOString()}] Logs cleared...\\n\`;
                document.getElementById('status-container').innerHTML = '<div class="status info">Logs cleared, ready for new tests...</div>';
            }
            
            updateSystemInfo() {
                document.getElementById('user-agent').textContent = navigator.userAgent;
                document.getElementById('react-status').textContent = typeof React !== 'undefined' ? '✅ Available' : '❌ Not Available';
                document.getElementById('arborist-status').textContent = typeof window.ReactArborist !== 'undefined' ? '✅ Available' : '❌ Not Available';
                document.getElementById('scribe-status').textContent = typeof window.ScribeFileTree !== 'undefined' ? '✅ Available' : '❌ Not Available';
            }
            
            generateMockFiles() {
                const files = [];
                const folders = ['src', 'tests', 'docs', 'utils', 'components', 'hooks', 'styles'];
                const extensions = ['.js', '.jsx', '.ts', '.tsx', '.css', '.md', '.json'];
                
                for (let i = 0; i < this.mockFileCount; i++) {
                    const folder = folders[i % folders.length];
                    const subfolder = i % 3 === 0 ? \`sub\${Math.floor(i / 3)}\` : '';
                    const ext = extensions[i % extensions.length];
                    const fileName = \`file\${i}\${ext}\`;
                    
                    const path = subfolder ? \`\${folder}/\${subfolder}/\${fileName}\` : \`\${folder}/\${fileName}\`;
                    files.push({ path });
                }
                
                return files;
            }
            
            async testTreeCreation() {
                this.log('Testing ScribeFileTree creation...', 'info');
                this.addStatus('Testing tree creation...', 'info');
                
                try {
                    if (typeof window.ScribeFileTree === 'undefined') {
                        throw new Error('ScribeFileTree not available globally');
                    }
                    
                    this.fileTree = new window.ScribeFileTree();
                    this.log('ScribeFileTree instance created successfully', 'success');
                    this.addStatus('✅ Tree creation successful', 'success');
                    return true;
                } catch (error) {
                    this.log(\`Tree creation failed: \${error.message}\`, 'error');
                    this.addStatus(\`❌ Tree creation failed: \${error.message}\`, 'error');
                    return false;
                }
            }
            
            testTreeBuilding() {
                this.log('Testing tree data building...', 'info');
                this.addStatus('Testing tree building...', 'info');
                
                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('❌ No tree instance for building test', 'error');
                    return false;
                }
                
                try {
                    const mockFiles = this.generateMockFiles();
                    const treeData = this.fileTree.buildTreeData(mockFiles);
                    
                    if (!Array.isArray(treeData) || treeData.length === 0) {
                        throw new Error('Tree building returned invalid data');
                    }
                    
                    this.log(\`Tree built successfully with \${treeData.length} root nodes\`, 'success');
                    this.addStatus(\`✅ Tree building successful (\${treeData.length} root nodes)\`, 'success');
                    return treeData;
                } catch (error) {
                    this.log(\`Tree building failed: \${error.message}\`, 'error');
                    this.addStatus(\`❌ Tree building failed: \${error.message}\`, 'error');
                    return false;
                }
            }
            
            testCheckboxes() {
                this.log('Testing checkbox functionality...', 'info');
                this.addStatus('Testing checkboxes...', 'info');
                
                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('❌ No tree instance for checkbox test', 'error');
                    return false;
                }
                
                try {
                    const selectedCount = this.fileTree.getSelectedFiles().length;
                    this.log(\`Current selected files: \${selectedCount}\`, 'info');
                    
                    // Test selection methods
                    this.fileTree.setSelectedFiles(['src/index.js', 'docs/README.md']);
                    const newSelectedCount = this.fileTree.getSelectedFiles().length;
                    
                    if (newSelectedCount === 2) {
                        this.log('Checkbox selection working correctly', 'success');
                        this.addStatus('✅ Checkbox functionality working', 'success');
                        return true;
                    } else {
                        throw new Error(\`Expected 2 selected files, got \${newSelectedCount}\`);
                    }
                } catch (error) {
                    this.log(\`Checkbox test failed: \${error.message}\`, 'error');
                    this.addStatus(\`❌ Checkbox test failed: \${error.message}\`, 'error');
                    return false;
                }
            }
            
            renderTree() {
                this.log('Attempting to render tree...', 'info');
                this.addStatus('Rendering tree...', 'info');
                
                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('❌ No tree instance for rendering', 'error');
                    return false;
                }
                
                try {
                    const mockFiles = this.generateMockFiles();
                    const success = this.fileTree.renderTree('tree-container', mockFiles);
                    
                    if (success) {
                        this.log('Tree rendered successfully', 'success');
                        this.addStatus('✅ Tree rendering successful', 'success');
                        
                        // Update system info to show current state
                        setTimeout(() => this.updateSystemInfo(), 1000);
                        return true;
                    } else {
                        throw new Error('Tree rendering returned false');
                    }
                } catch (error) {
                    this.log(\`Tree rendering failed: \${error.message}\`, 'error');
                    this.addStatus(\`❌ Tree rendering failed: \${error.message}\`, 'error');
                    return false;
                }
            }
            
            async runAllTests() {
                this.log('Running all debug tests...', 'info');
                this.addStatus('Running comprehensive test suite...', 'info');
                
                const results = [];
                
                results.push(await this.testTreeCreation());
                results.push(this.testTreeBuilding());
                results.push(this.testCheckboxes());
                results.push(this.renderTree());
                
                const passed = results.filter(r => r === true).length;
                const total = results.length;
                
                if (passed === total) {
                    this.log(\`All tests passed! (\${passed}/\${total})\`, 'success');
                    this.addStatus(\`🎉 All tests passed! (\${passed}/\${total})\`, 'success');
                } else {
                    this.log(\`Some tests failed (\${passed}/\${total} passed)\`, 'warning');
                    this.addStatus(\`⚠️ Some tests failed (\${passed}/\${total} passed)\`, 'warning');
                }
            }
            
            updateFileCount(value) {
                this.mockFileCount = parseInt(value);
                document.getElementById('file-count-display').textContent = value;
                this.log(\`Updated mock file count to \${value}\`, 'info');
            }
            
            updateMaxDepth(value) {
                this.maxDepth = parseInt(value);
                document.getElementById('max-depth-display').textContent = value;
                this.log(\`Updated max depth to \${value}\`, 'info');
            }
        }
        
        window.debugTests = new DebugTests();
        
        // Auto-update system info when ScribeFileTree becomes available
        const checkForScribeFileTree = () => {
            if (typeof window.ScribeFileTree !== 'undefined') {
                window.debugTests.updateSystemInfo();
                window.debugTests.log('ScribeFileTree detected!', 'success');
            } else {
                setTimeout(checkForScribeFileTree, 1000);
            }
        };
        checkForScribeFileTree();
    </script>
    
    <!-- Load the ScribeFileTree bundle -->
    <script type="module" src="./scribe-tree-bundle.js"></script>
</body>
</html>`;

    try {
      const testFilePath = join(process.cwd(), 'arborist-debug-test.html');
      writeFileSync(testFilePath, template);
      this.addFinding('template', `Test template generated: ${testFilePath}`);
      log.success(`Debug test template created: ${testFilePath}`);
      return testFilePath;
    } catch (error) {
      this.addIssue('error', 'template', `Failed to generate test template: ${error.message}`);
      return false;
    }
  }

  // Run all debugging tests
  async runAllTests() {
    log.section('React Arborist Debug Session');
    log.info(`Starting comprehensive debugging at ${new Date().toISOString()}`);
    
    // Run all test methods
    const fileTree = await this.testScribeFileTreeCreation();
    if (fileTree) {
      const treeData = this.testTreeDataBuilding(fileTree);
      if (treeData) {
        this.testCheckboxManagement(fileTree, treeData);
      }
      this.testComponentCreation(fileTree);
      this.testTreeRendering(fileTree);
    }
    
    this.checkTemplateIntegration();
    this.generateTestTemplate();
    
    this.generateReport();
  }

  // Count issues by severity
  countIssuesBySeverity() {
    return this.issues.reduce((acc, issue) => {
      acc[issue.severity] = (acc[issue.severity] || 0) + 1;
      return acc;
    }, {});
  }

  // Display issue counts by severity
  displayIssueCounts(issueCounts) {
    Object.entries(issueCounts).forEach(([severity, count]) => {
      const logFn = severity === 'error' ? 'error' : severity === 'warning' ? 'warning' : 'info';
      log[logFn](`  ${severity.toUpperCase()}: ${count}`);
    });
  }

  // Display detailed issues
  displayIssues() {
    if (this.issues.length === 0) return;

    log.subsection('Issues Found');
    this.issues.forEach((issue) => {
      const symbol = issue.severity === 'error' ? '❌' : issue.severity === 'warning' ? '⚠️' : 'ℹ️';
      log[issue.severity](`${symbol} [${issue.category.toUpperCase()}] ${issue.message}`);
      if (issue.details) {
        log.debug(`    Details: ${JSON.stringify(issue.details, null, 2)}`);
      }
    });
  }

  // Display findings
  displayFindings() {
    if (this.findings.length === 0) return;

    log.subsection('Key Findings');
    this.findings.forEach((finding) => {
      log.success(`✅ [${finding.category.toUpperCase()}] ${finding.message}`);
    });
  }

  // Display recommendations
  displayRecommendations() {
    if (this.recommendations.length === 0) return;

    log.subsection('Recommendations');
    this.recommendations.forEach((rec) => {
      const priority = rec.priority === 'high' ? '🔴' : rec.priority === 'medium' ? '🟡' : '🟢';
      log.info(`${priority} ${rec.message}`);
      if (rec.action) {
        log.debug(`    Action: ${rec.action}`);
      }
    });
  }

  // Save report data to file
  saveReportToFile(reportData) {
    try {
      const reportPath = join(process.cwd(), 'arborist-debug-report.json');
      writeFileSync(reportPath, JSON.stringify(reportData, null, 2));
      log.success(`Detailed report saved to: ${reportPath}`);
    } catch (error) {
      log.error(`Failed to save report: ${error.message}`);
    }
  }

  // Display next steps based on issue severity
  displayNextSteps() {
    if (this.issues.some(i => i.severity === 'error')) {
      log.warning('Critical issues found. Address errors before proceeding with integration.');
    } else if (this.issues.some(i => i.severity === 'warning')) {
      log.info('Some warnings found. Review and address as needed.');
    } else {
      log.success('No critical issues found! Component should integrate successfully.');
    }
  }

  // Generate comprehensive debug report
  generateReport() {
    log.section('Debug Report Summary');

    const issueCounts = this.countIssuesBySeverity();

    log.info(`Total Issues Found: ${this.issues.length}`);
    this.displayIssueCounts(issueCounts);
    log.info(`Findings: ${this.findings.length}`);
    log.info(`Recommendations: ${this.recommendations.length}`);

    this.displayIssues();
    this.displayFindings();
    this.displayRecommendations();

    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        totalIssues: this.issues.length,
        issueCounts,
        totalFindings: this.findings.length,
        totalRecommendations: this.recommendations.length
      },
      issues: this.issues,
      findings: this.findings,
      recommendations: this.recommendations
    };

    this.saveReportToFile(reportData);

    log.section('Debug Session Complete');
    this.displayNextSteps();
  }
}

// Main execution
async function main() {
  const arboristDebugger = new ArboristDebugger();
  await arboristDebugger.runAllTests();
}

// Run if this file is executed directly
if (import.meta.main) {
  main().catch(error => {
    console.error('Debug script failed:', error);
    process.exit(1);
  });
}

export default ArboristDebugger;