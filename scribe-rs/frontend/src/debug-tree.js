#!/usr/bin/env bun
// Standalone debugging script for React Arborist component integration
// Run with: bun run src/debug-tree.js

import {
  colors,
  log,
  mockFiles,
  setupMockDOMEnvironment,
  buildNodeMapFromTreeData,
  logTreeStructure,
  saveReportToFile,
  displayIssueCounts,
  displayNextSteps,
  generateTestTemplate
} from './debug-tree-helpers.js';

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
      setupMockDOMEnvironment();

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

  // Validate tree data array
  validateTreeDataArray(treeData) {
    if (!Array.isArray(treeData)) {
      this.addIssue('error', 'tree-building', 'buildTreeData did not return an array');
      return false;
    }

    if (treeData.length === 0) {
      this.addIssue('warning', 'tree-building', 'buildTreeData returned empty array');
      return false;
    }

    return true;
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

      if (!this.validateTreeDataArray(treeData)) {
        return false;
      }

      treeData.forEach((node, index) => {
        this.validateTreeNode(node, `root[${index}]`);
      });

      this.addFinding('tree-building', `Successfully built tree with ${treeData.length} root nodes`);
      log.success(`Tree data built successfully with ${treeData.length} root nodes`);

      log.debug('Tree structure:');
      logTreeStructure(treeData);

      return treeData;

    } catch (error) {
      this.addIssue('error', 'tree-building', `Tree building failed: ${error.message}`, { stack: error.stack });
      log.error(`Tree building failed: ${error.message}`);
      return false;
    }
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
      const nodeMap = buildNodeMapFromTreeData(treeData);

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

  // Validate a component factory function
  validateComponentFactory(factoryFn, factoryName, componentName) {
    const Component = factoryFn();
    if (typeof Component !== 'function') {
      this.addIssue('error', 'component', `${factoryName} did not return a function`);
      return false;
    }
    this.addFinding('component', `${componentName} created successfully`);
    return true;
  }

  // Test React component creation
  testComponentCreation(fileTree) {
    log.subsection('Testing React Component Creation');

    if (!fileTree) {
      this.addIssue('error', 'component', 'No fileTree instance for component testing');
      return false;
    }

    try {
      this.validateComponentFactory(() => fileTree.createNodeComponent(), 'createNodeComponent', 'Node component');
      this.validateComponentFactory(() => fileTree.createTreeComponent(), 'createTreeComponent', 'Tree component');

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

  // Test rendering with a missing container
  testMissingContainerRendering(fileTree) {
    const result = fileTree.renderTree('non-existent-container', mockFiles);
    if (result !== false) {
      this.addIssue('warning', 'rendering', 'renderTree should return false for missing container');
    } else {
      this.addFinding('rendering', 'Correctly handles missing container');
    }
  }

  // Test rendering with a mock container
  testMockContainerRendering(fileTree) {
    let renderCalled = false;
    global.window.ReactDOM.createRoot = () => ({
      render: () => {
        renderCalled = true;
        log.debug('Mock render method called');
      }
    });

    const result = fileTree.renderTree('mock-container', mockFiles);
    if (result && renderCalled) {
      this.addFinding('rendering', 'Tree rendering simulation successful');
      log.success('Tree rendering test passed');
    } else {
      this.addIssue('warning', 'rendering', 'Tree rendering simulation issues detected');
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
      this.testMissingContainerRendering(fileTree);
      this.testMockContainerRendering(fileTree);
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
    generateTestTemplate(this.addFinding.bind(this), this.addIssue.bind(this));
    
    this.generateReport();
  }

  // Count issues by severity
  countIssuesBySeverity() {
    return this.issues.reduce((acc, issue) => {
      acc[issue.severity] = (acc[issue.severity] || 0) + 1;
      return acc;
    }, {});
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

  // Generate comprehensive debug report
  generateReport() {
    log.section('Debug Report Summary');

    const issueCounts = this.countIssuesBySeverity();

    log.info(`Total Issues Found: ${this.issues.length}`);
    displayIssueCounts(issueCounts);
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

    saveReportToFile(reportData);

    log.section('Debug Session Complete');
    displayNextSteps(this.issues);
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