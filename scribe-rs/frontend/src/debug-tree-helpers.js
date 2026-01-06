// Helper utilities for debug-tree.js
import { writeFileSync } from 'fs';
import { join } from 'path';

// Import the HTML template
import { getDebugTemplate } from './debug-tree-template.js';

// ANSI color codes for terminal output
export const colors = {
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

export const log = {
  info: (msg) => console.log(`${colors.blue}i${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}v${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}!${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}x${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.bold}${colors.cyan}=== ${msg} ===${colors.reset}`),
  subsection: (msg) => console.log(`\n${colors.bold}--- ${msg} ---${colors.reset}`),
  debug: (msg) => console.log(`${colors.dim}[DEBUG]${colors.reset} ${msg}`)
};

// Mock file data for testing
export const mockFiles = [
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

// Setup mock DOM environment for testing
export function setupMockDOMEnvironment() {
  global.document = {
    getElementById: (id) => {
      log.debug(`Mock DOM: getElementById called with '${id}'`);
      return {
        scrollIntoView: () => log.debug('Mock scrollIntoView called'),
        innerHTML: '',
        appendChild: () => log.debug('Mock appendChild called'),
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
}

// Build a node map from tree data for testing
export function buildNodeMapFromTreeData(treeData) {
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

// Log tree structure recursively
export function logTreeStructure(nodes, indent = '') {
  nodes.forEach(node => {
    log.debug(`${indent}${node.isFolder ? 'D' : 'F'} ${node.name} (${node.path})`);
    if (node.children) {
      logTreeStructure(node.children, indent + '  ');
    }
  });
}

// Save report data to file
export function saveReportToFile(reportData) {
  try {
    const reportPath = join(process.cwd(), 'arborist-debug-report.json');
    writeFileSync(reportPath, JSON.stringify(reportData, null, 2));
    log.success(`Detailed report saved to: ${reportPath}`);
  } catch (error) {
    log.error(`Failed to save report: ${error.message}`);
  }
}

// Display issue counts by severity
export function displayIssueCounts(issueCounts) {
  Object.entries(issueCounts).forEach(([severity, count]) => {
    const logFn = severity === 'error' ? 'error' : severity === 'warning' ? 'warning' : 'info';
    log[logFn](`  ${severity.toUpperCase()}: ${count}`);
  });
}

// Display next steps based on issue severity
export function displayNextSteps(issues) {
  if (issues.some(i => i.severity === 'error')) {
    log.warning('Critical issues found. Address errors before proceeding with integration.');
  } else if (issues.some(i => i.severity === 'warning')) {
    log.info('Some warnings found. Review and address as needed.');
  } else {
    log.success('No critical issues found! Component should integrate successfully.');
  }
}

// Generate HTML test template for visual debugging
export function generateTestTemplate(addFinding, addIssue) {
  log.subsection('Generating HTML Test Template');

  try {
    const template = getDebugTemplate();
    const testFilePath = join(process.cwd(), 'arborist-debug-test.html');
    writeFileSync(testFilePath, template);
    addFinding('template', `Test template generated: ${testFilePath}`);
    log.success(`Debug test template created: ${testFilePath}`);
    return testFilePath;
  } catch (error) {
    addIssue('error', 'template', `Failed to generate test template: ${error.message}`);
    return false;
  }
}
