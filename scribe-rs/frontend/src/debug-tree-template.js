// HTML template for the debug test page
export function getDebugTemplate() {
  return `<!DOCTYPE html>
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
            <h1>React Arborist Debug Test</h1>
            <p>Debugging ScribeFileTree component integration</p>
        </div>

        <div class="debug-content">
            <div class="debug-section">
                <h3>Debug Controls</h3>
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
                <h3>Test Status</h3>
                <div id="status-container">
                    <div class="status info">Ready to run tests...</div>
                </div>
            </div>

            <div class="debug-section">
                <h3>Tree Container</h3>
                <div id="tree-container">
                    Tree will render here...
                </div>
            </div>

            <div class="debug-section">
                <h3>Debug Logs</h3>
                <div id="log-output" class="log-output">
[${new Date().toISOString()}] Debug environment ready...
                </div>
            </div>

            <div class="debug-section">
                <h3>System Information</h3>
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
                document.getElementById('react-status').textContent = typeof React !== 'undefined' ? 'Available' : 'Not Available';
                document.getElementById('arborist-status').textContent = typeof window.ReactArborist !== 'undefined' ? 'Available' : 'Not Available';
                document.getElementById('scribe-status').textContent = typeof window.ScribeFileTree !== 'undefined' ? 'Available' : 'Not Available';
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
                    this.addStatus('Tree creation successful', 'success');
                    return true;
                } catch (error) {
                    this.log(\`Tree creation failed: \${error.message}\`, 'error');
                    this.addStatus(\`Tree creation failed: \${error.message}\`, 'error');
                    return false;
                }
            }

            testTreeBuilding() {
                this.log('Testing tree data building...', 'info');
                this.addStatus('Testing tree building...', 'info');

                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('No tree instance for building test', 'error');
                    return false;
                }

                try {
                    const mockFiles = this.generateMockFiles();
                    const treeData = this.fileTree.buildTreeData(mockFiles);

                    if (!Array.isArray(treeData) || treeData.length === 0) {
                        throw new Error('Tree building returned invalid data');
                    }

                    this.log(\`Tree built successfully with \${treeData.length} root nodes\`, 'success');
                    this.addStatus(\`Tree building successful (\${treeData.length} root nodes)\`, 'success');
                    return treeData;
                } catch (error) {
                    this.log(\`Tree building failed: \${error.message}\`, 'error');
                    this.addStatus(\`Tree building failed: \${error.message}\`, 'error');
                    return false;
                }
            }

            testCheckboxes() {
                this.log('Testing checkbox functionality...', 'info');
                this.addStatus('Testing checkboxes...', 'info');

                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('No tree instance for checkbox test', 'error');
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
                        this.addStatus('Checkbox functionality working', 'success');
                        return true;
                    } else {
                        throw new Error(\`Expected 2 selected files, got \${newSelectedCount}\`);
                    }
                } catch (error) {
                    this.log(\`Checkbox test failed: \${error.message}\`, 'error');
                    this.addStatus(\`Checkbox test failed: \${error.message}\`, 'error');
                    return false;
                }
            }

            renderTree() {
                this.log('Attempting to render tree...', 'info');
                this.addStatus('Rendering tree...', 'info');

                if (!this.fileTree) {
                    this.log('No fileTree instance available', 'error');
                    this.addStatus('No tree instance for rendering', 'error');
                    return false;
                }

                try {
                    const mockFiles = this.generateMockFiles();
                    const success = this.fileTree.renderTree('tree-container', mockFiles);

                    if (success) {
                        this.log('Tree rendered successfully', 'success');
                        this.addStatus('Tree rendering successful', 'success');

                        // Update system info to show current state
                        setTimeout(() => this.updateSystemInfo(), 1000);
                        return true;
                    } else {
                        throw new Error('Tree rendering returned false');
                    }
                } catch (error) {
                    this.log(\`Tree rendering failed: \${error.message}\`, 'error');
                    this.addStatus(\`Tree rendering failed: \${error.message}\`, 'error');
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
                    this.addStatus(\`All tests passed! (\${passed}/\${total})\`, 'success');
                } else {
                    this.log(\`Some tests failed (\${passed}/\${total} passed)\`, 'warning');
                    this.addStatus(\`Some tests failed (\${passed}/\${total} passed)\`, 'warning');
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
}
