// Practical debugging utilities for React Arborist component troubleshooting
// Focused on real-world debugging scenarios without complex dependencies

/**
 * Simple console debugging utility with categorized logging
 */
export class SimpleDebugger {
  constructor(enabled = true) {
    this.enabled = enabled;
    this.logs = [];
    this.startTime = Date.now();
  }

  log(message, category = 'INFO', data = null) {
    if (!this.enabled) return;
    
    const timestamp = Date.now() - this.startTime;
    const logEntry = {
      timestamp,
      category,
      message,
      data
    };
    
    this.logs.push(logEntry);
    
    // Simple console output with color coding
    const colors = {
      'ERROR': '\x1b[31m',
      'WARN': '\x1b[33m', 
      'SUCCESS': '\x1b[32m',
      'INFO': '\x1b[36m',
      'DEBUG': '\x1b[37m'
    };
    
    const color = colors[category] || '\x1b[37m';
    const reset = '\x1b[0m';
    
    console.log(`${color}[${category}] +${timestamp}ms: ${message}${reset}`);
    if (data) {
      console.log('  Data:', data);
    }
  }

  error(message, data) { this.log(message, 'ERROR', data); }
  warn(message, data) { this.log(message, 'WARN', data); }
  success(message, data) { this.log(message, 'SUCCESS', data); }
  info(message, data) { this.log(message, 'INFO', data); }
  debug(message, data) { this.log(message, 'DEBUG', data); }

  getLogs() { return this.logs; }
  clear() { this.logs = []; this.startTime = Date.now(); }
}

/**
 * Tree data structure validator
 */
export const validateTreeStructure = (treeData) => {
  const issues = [];
  const warnings = [];
  
  if (!Array.isArray(treeData)) {
    issues.push('Tree data must be an array');
    return { valid: false, issues, warnings };
  }

  const validateNode = (node, path = '', depth = 0) => {
    // Required fields
    if (!node.id) issues.push(`Missing 'id' at ${path}`);
    if (!node.name) issues.push(`Missing 'name' at ${path}`);
    if (typeof node.isFolder !== 'boolean') issues.push(`Missing 'isFolder' at ${path}`);
    
    // Structure validation
    if (node.isFolder) {
      if (!Array.isArray(node.children)) {
        warnings.push(`Folder at ${path} missing children array`);
      } else {
        node.children.forEach((child, index) => {
          validateNode(child, `${path}[${index}]`, depth + 1);
        });
      }
    } else {
      if (node.children) {
        warnings.push(`File at ${path} has children array (should be undefined)`);
      }
    }
    
    // Depth warning
    if (depth > 10) {
      warnings.push(`Very deep nesting at ${path} (depth: ${depth})`);
    }
  };

  treeData.forEach((node, index) => {
    validateNode(node, `root[${index}]`);
  });

  return {
    valid: issues.length === 0,
    issues,
    warnings,
    stats: {
      rootNodes: treeData.length,
      totalIssues: issues.length,
      totalWarnings: warnings.length
    }
  };
};

/**
 * Checkbox state consistency checker
 */
export const validateCheckboxStates = (checkboxStates, selectedFiles) => {
  const issues = [];
  
  // Check that all selected files have correct checkbox state
  selectedFiles.forEach(filePath => {
    const state = checkboxStates.get(filePath);
    if (!state) {
      issues.push(`Selected file ${filePath} missing checkbox state`);
    } else if (!state.checked) {
      issues.push(`Selected file ${filePath} has unchecked state`);
    }
  });
  
  // Check that all checked states are in selected files
  checkboxStates.forEach((state, path) => {
    if (state.checked && !selectedFiles.has(path)) {
      // This might be a folder, so only report for files
      if (!path.endsWith('/')) {
        issues.push(`Checked path ${path} not in selected files`);
      }
    }
  });
  
  return {
    consistent: issues.length === 0,
    issues,
    stats: {
      totalStates: checkboxStates.size,
      selectedFiles: selectedFiles.size,
      checkedStates: Array.from(checkboxStates.values()).filter(s => s.checked).length,
      indeterminateStates: Array.from(checkboxStates.values()).filter(s => s.indeterminate).length
    }
  };
};

/**
 * DOM integration checker
 */
export const checkDOMIntegration = () => {
  const results = {
    environment: 'unknown',
    issues: [],
    warnings: [],
    dependencies: {}
  };
  
  // Detect environment
  if (typeof window !== 'undefined') {
    results.environment = 'browser';
    
    // Check for required globals
    results.dependencies.React = typeof window.React !== 'undefined';
    results.dependencies.ReactDOM = typeof window.ReactDOM !== 'undefined';
    results.dependencies.ReactArborist = typeof window.ReactArborist !== 'undefined';
    results.dependencies.LucideReact = typeof window.LucideReact !== 'undefined';
    results.dependencies.ScribeFileTree = typeof window.ScribeFileTree !== 'undefined';
    
    // Check DOM capabilities
    if (typeof document !== 'undefined') {
      results.dependencies.document = true;
      results.dependencies.getElementById = typeof document.getElementById === 'function';
      results.dependencies.createElement = typeof document.createElement === 'function';
    } else {
      results.issues.push('document object not available');
    }
    
    // Report missing dependencies
    Object.entries(results.dependencies).forEach(([name, available]) => {
      if (!available) {
        if (name === 'ScribeFileTree') {
          results.warnings.push(`${name} not available (may be expected if not loaded yet)`);
        } else {
          results.issues.push(`${name} not available`);
        }
      }
    });
    
  } else if (typeof global !== 'undefined') {
    results.environment = 'node';
    results.warnings.push('Running in Node.js environment - DOM features not available');
  } else {
    results.environment = 'unknown';
    results.issues.push('Unknown JavaScript environment');
  }
  
  return results;
};

/**
 * Performance timing utility
 */
export class PerformanceTimer {
  constructor() {
    this.timers = new Map();
    this.results = new Map();
  }
  
  start(label) {
    this.timers.set(label, Date.now());
  }
  
  end(label) {
    const startTime = this.timers.get(label);
    if (!startTime) {
      console.warn(`Timer '${label}' was not started`);
      return null;
    }
    
    const duration = Date.now() - startTime;
    this.results.set(label, duration);
    this.timers.delete(label);
    
    return duration;
  }
  
  getResult(label) {
    return this.results.get(label);
  }
  
  getAllResults() {
    return Object.fromEntries(this.results);
  }
  
  report() {
    console.log('Performance Report:');
    this.results.forEach((duration, label) => {
      const color = duration > 1000 ? '\x1b[31m' : duration > 100 ? '\x1b[33m' : '\x1b[32m';
      console.log(`  ${color}${label}: ${duration}ms\x1b[0m`);
    });
  }
}

/**
 * Tree rendering troubleshooter
 */
export const diagnoseRenderingIssues = (containerId, fileData) => {
  const diagnosis = {
    canRender: true,
    issues: [],
    warnings: [],
    recommendations: []
  };
  
  // Check container element
  if (typeof document === 'undefined') {
    diagnosis.canRender = false;
    diagnosis.issues.push('document object not available');
    return diagnosis;
  }
  
  const container = document.getElementById(containerId);
  if (!container) {
    diagnosis.canRender = false;
    diagnosis.issues.push(`Container element '${containerId}' not found`);
    diagnosis.recommendations.push(`Ensure an element with id='${containerId}' exists in the DOM`);
  } else {
    // Check container properties
    if (container.style.display === 'none') {
      diagnosis.warnings.push('Container element is hidden (display: none)');
    }
    if (container.offsetWidth === 0 || container.offsetHeight === 0) {
      diagnosis.warnings.push('Container element has zero dimensions');
    }
  }
  
  // Check file data
  if (!fileData || !Array.isArray(fileData)) {
    diagnosis.canRender = false;
    diagnosis.issues.push('Invalid file data provided');
  } else if (fileData.length === 0) {
    diagnosis.warnings.push('Empty file data provided');
  }
  
  // Check React dependencies
  if (typeof window.React === 'undefined') {
    diagnosis.canRender = false;
    diagnosis.issues.push('React not available');
    diagnosis.recommendations.push('Load React before initializing ScribeFileTree');
  }
  
  if (typeof window.ReactDOM === 'undefined') {
    diagnosis.canRender = false;
    diagnosis.issues.push('ReactDOM not available');
    diagnosis.recommendations.push('Load ReactDOM before initializing ScribeFileTree');
  }
  
  if (typeof window.ReactArborist === 'undefined') {
    diagnosis.canRender = false;
    diagnosis.issues.push('React Arborist not available');
    diagnosis.recommendations.push('Load React Arborist library before initializing ScribeFileTree');
  }
  
  return diagnosis;
};

/**
 * Tree state inspector
 */
export const inspectTreeState = (fileTree) => {
  if (!fileTree) {
    return { error: 'No fileTree instance provided' };
  }
  
  const state = {
    checkboxStates: {
      total: fileTree.checkboxStates?.size || 0,
      checked: 0,
      indeterminate: 0
    },
    selectedFiles: {
      total: fileTree.selectedFiles?.size || 0,
      list: []
    },
    components: {
      nodeComponent: typeof fileTree.createNodeComponent === 'function',
      treeComponent: typeof fileTree.createTreeComponent === 'function',
      initialized: typeof fileTree.FileTreeComponent === 'function'
    }
  };
  
  // Analyze checkbox states
  if (fileTree.checkboxStates) {
    fileTree.checkboxStates.forEach(stateObj => {
      if (stateObj.checked) state.checkboxStates.checked++;
      if (stateObj.indeterminate) state.checkboxStates.indeterminate++;
    });
  }
  
  // Get selected files list
  if (fileTree.selectedFiles) {
    state.selectedFiles.list = Array.from(fileTree.selectedFiles);
  }
  
  return state;
};

/**
 * Create a simple debugging wrapper for ScribeFileTree
 */
export const createDebugWrapper = (ScribeFileTree, options = {}) => {
  const simpleDebugger = new SimpleDebugger(options.enabled !== false);
  
  return class DebugScribeFileTree extends ScribeFileTree {
    constructor() {
      super();
      simpleDebugger.info('ScribeFileTree instance created');
    }
    
    buildTreeData(files) {
      simpleDebugger.info(`Building tree data from ${files?.length || 0} files`);
      const timer = new PerformanceTimer();
      timer.start('buildTreeData');
      
      const result = super.buildTreeData(files);
      const duration = timer.end('buildTreeData');
      
      simpleDebugger.success(`Tree built successfully in ${duration}ms`, {
        rootNodes: result?.length || 0,
        totalFiles: files?.length || 0
      });
      
      return result;
    }
    
    toggleFileCheckbox(nodeMap, path, isFile) {
      simpleDebugger.debug(`Toggling checkbox for ${isFile ? 'file' : 'folder'}: ${path}`);
      
      const result = super.toggleFileCheckbox(nodeMap, path, isFile);
      
      simpleDebugger.debug(`Checkbox toggled`, {
        path,
        isFile,
        selectedCount: this.selectedFiles.size
      });
      
      return result;
    }
    
    renderTree(containerId, fileData) {
      simpleDebugger.info(`Attempting to render tree in container: ${containerId}`);
      
      const diagnosis = diagnoseRenderingIssues(containerId, fileData);
      if (!diagnosis.canRender) {
        simpleDebugger.error('Cannot render tree', diagnosis);
        return false;
      }
      
      if (diagnosis.warnings.length > 0) {
        simpleDebugger.warn('Rendering warnings detected', diagnosis.warnings);
      }
      
      const timer = new PerformanceTimer();
      timer.start('renderTree');
      
      const result = super.renderTree(containerId, fileData);
      const duration = timer.end('renderTree');
      
      if (result) {
        simpleDebugger.success(`Tree rendered successfully in ${duration}ms`);
      } else {
        simpleDebugger.error(`Tree rendering failed after ${duration}ms`);
      }
      
      return result;
    }
    
    getDebugInfo() {
      return {
        logs: simpleDebugger.getLogs(),
        state: inspectTreeState(this),
        performance: timer.getAllResults()
      };
    }
    
    clearDebugLogs() {
      simpleDebugger.clear();
    }
  };
};

/**
 * Quick diagnostic function for troubleshooting
 */
export const quickDiagnosis = () => {
  console.log('🔍 React Arborist Quick Diagnosis');
  console.log('================================');
  
  const dom = checkDOMIntegration();
  console.log(`Environment: ${dom.environment}`);
  
  if (dom.issues.length > 0) {
    console.log('❌ Issues found:');
    dom.issues.forEach(issue => console.log(`  - ${issue}`));
  }
  
  if (dom.warnings.length > 0) {
    console.log('⚠️ Warnings:');
    dom.warnings.forEach(warning => console.log(`  - ${warning}`));
  }
  
  if (dom.dependencies.ScribeFileTree) {
    const state = inspectTreeState(window.ScribeFileTree);
    console.log('📊 ScribeFileTree State:');
    console.log(`  - Checkbox states: ${state.checkboxStates.total}`);
    console.log(`  - Selected files: ${state.selectedFiles.total}`);
    console.log(`  - Components initialized: ${state.components.initialized}`);
  }
  
  return { dom };
};

// Global debug instance for quick access
export const simpleDebugger = new SimpleDebugger();

// Make utilities available globally for browser console access
if (typeof window !== 'undefined') {
  window.ScribeDebug = {
    quickDiagnosis,
    validateTreeStructure,
    validateCheckboxStates,
    checkDOMIntegration,
    diagnoseRenderingIssues,
    inspectTreeState,
    simpleDebugger,
    PerformanceTimer,
    SimpleDebugger,
    createDebugWrapper
  };
}