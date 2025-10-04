// Bun-native test suite for React Arborist Tree component
import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import React from 'react';

// Mock data generator for testing
const createMockTreeData = (depth = 2, breadth = 3) => {
  const nodes = [];
  
  for (let i = 0; i < breadth; i++) {
    const node = {
      id: `node-${i}`,
      name: `Node ${i}`,
      path: `node-${i}`,
      isFolder: depth > 0,
      type: depth > 0 ? 'folder' : 'file'
    };
    
    if (depth > 0) {
      node.children = createMockTreeData(depth - 1, Math.max(1, breadth - 1));
    }
    
    nodes.push(node);
  }
  
  return nodes;
};

// Mock file data similar to what ScribeFileTree expects
const createMockFileData = () => [
  { path: 'src/index.js' },
  { path: 'src/components/TreeView.jsx' },
  { path: 'src/components/Button.jsx' },
  { path: 'src/utils/helpers.js' },
  { path: 'tests/unit/component.test.js' },
  { path: 'tests/integration/app.test.js' },
  { path: 'README.md' },
  { path: 'package.json' }
];

describe('ScribeFileTree Class', () => {
  let ScribeFileTree;
  let fileTree;
  
  beforeEach(async () => {
    // Mock DOM elements
    global.document = {
      getElementById: (id) => ({
        scrollIntoView: () => {},
        innerHTML: '',
        appendChild: () => {},
        style: {}
      }),
      createElement: (tag) => ({
        className: '',
        style: {},
        appendChild: () => {},
        setAttribute: () => {},
        getAttribute: () => null,
        addEventListener: () => {},
        removeEventListener: () => {}
      })
    };
    
    global.window = {
      React: React,
      ReactDOM: { createRoot: () => ({ render: () => {} }) }
    };
    
    global.console = {
      log: () => {},
      warn: () => {},
      error: () => {},
      groupCollapsed: () => {},
      groupEnd: () => {},
      trace: () => {},
      clear: () => {}
    };
    
    // Import ScribeFileTree after mocking globals
    const module = await import('../index.js');
    ScribeFileTree = module.default;
    fileTree = new ScribeFileTree();
  });

  describe('Tree Data Building', () => {
    test('should build hierarchical tree from flat file paths', () => {
      const files = createMockFileData();
      const treeData = fileTree.buildTreeData(files);
      
      expect(Array.isArray(treeData)).toBe(true);
      expect(treeData.length).toBeGreaterThan(0);
      
      // Should have root level folders and files
      const srcFolder = treeData.find(node => node.name === 'src');
      expect(srcFolder).toBeDefined();
      expect(srcFolder.isFolder).toBe(true);
      expect(Array.isArray(srcFolder.children)).toBe(true);
      expect(srcFolder.children.length).toBeGreaterThan(0);
    });

    test('should handle empty file array', () => {
      const treeData = fileTree.buildTreeData([]);
      expect(Array.isArray(treeData)).toBe(true);
      expect(treeData.length).toBe(0);
    });

    test('should handle null/undefined files', () => {
      const treeData1 = fileTree.buildTreeData(null);
      const treeData2 = fileTree.buildTreeData(undefined);
      
      expect(Array.isArray(treeData1)).toBe(true);
      expect(Array.isArray(treeData2)).toBe(true);
      expect(treeData1.length).toBe(0);
      expect(treeData2.length).toBe(0);
    });

    test('should correctly set file properties', () => {
      const files = [{ path: 'test.js' }];
      const treeData = fileTree.buildTreeData(files);
      
      expect(treeData[0].name).toBe('test.js');
      expect(treeData[0].isFolder).toBe(false);
      expect(treeData[0].path).toBe('test.js');
      expect(treeData[0].fileIndex).toBe(0);
    });

    test('should create proper folder hierarchy', () => {
      const files = [
        { path: 'src/components/Button.jsx' },
        { path: 'src/utils/helper.js' }
      ];
      const treeData = fileTree.buildTreeData(files);
      
      const srcFolder = treeData.find(node => node.name === 'src');
      expect(srcFolder.isFolder).toBe(true);
      expect(srcFolder.children).toBeDefined();
      
      const componentsFolder = srcFolder.children.find(node => node.name === 'components');
      expect(componentsFolder.isFolder).toBe(true);
      
      const buttonFile = componentsFolder.children.find(node => node.name === 'Button.jsx');
      expect(buttonFile.isFolder).toBe(false);
    });
  });

  describe('Checkbox State Management', () => {
    let nodeMap;
    
    beforeEach(() => {
      const files = createMockFileData();
      const treeData = fileTree.buildTreeData(files);
      
      // Build node map
      nodeMap = new Map();
      const buildNodeMap = (nodes) => {
        nodes.forEach(node => {
          nodeMap.set(node.path, node);
          if (node.children) {
            buildNodeMap(node.children);
          }
        });
      };
      buildNodeMap(treeData);
    });

    test('should initialize checkbox states for all nodes', () => {
      expect(fileTree.checkboxStates.size).toBeGreaterThan(0);
      
      // Check that all nodes have initial state
      nodeMap.forEach((node, path) => {
        const state = fileTree.checkboxStates.get(path);
        expect(state).toBeDefined();
        expect(state.checked).toBe(false);
        expect(state.indeterminate).toBe(false);
      });
    });

    test('should toggle file checkbox state', () => {
      const filePath = 'src/index.js';
      const initialState = fileTree.checkboxStates.get(filePath);
      
      expect(initialState.checked).toBe(false);
      expect(fileTree.selectedFiles.has(filePath)).toBe(false);
      
      fileTree.toggleFileCheckbox(nodeMap, filePath, true);
      
      const newState = fileTree.checkboxStates.get(filePath);
      expect(newState.checked).toBe(true);
      expect(fileTree.selectedFiles.has(filePath)).toBe(true);
    });

    test('should update parent folder states when file is selected', () => {
      const filePath = 'src/components/TreeView.jsx';
      fileTree.toggleFileCheckbox(nodeMap, filePath, true);
      
      // Parent folders should show indeterminate state
      const componentsState = fileTree.checkboxStates.get('src/components');
      const srcState = fileTree.checkboxStates.get('src');
      
      expect(componentsState.indeterminate).toBe(true);
      expect(srcState.indeterminate).toBe(true);
    });

    test('should handle folder checkbox toggle', () => {
      const folderPath = 'src/components';
      fileTree.toggleFileCheckbox(nodeMap, folderPath, false);
      
      const folderState = fileTree.checkboxStates.get(folderPath);
      expect(folderState.checked).toBe(true);
      
      // All children should be selected
      const folder = nodeMap.get(folderPath);
      if (folder && folder.children) {
        folder.children.forEach(child => {
          const childState = fileTree.checkboxStates.get(child.path);
          expect(childState.checked).toBe(true);
          if (!child.isFolder) {
            expect(fileTree.selectedFiles.has(child.path)).toBe(true);
          }
        });
      }
    });

    test('should clear all selections', () => {
      // Select some files first
      fileTree.toggleFileCheckbox(nodeMap, 'src/index.js', true);
      fileTree.toggleFileCheckbox(nodeMap, 'README.md', true);
      
      expect(fileTree.selectedFiles.size).toBe(2);
      
      fileTree.clearSelection();
      
      expect(fileTree.selectedFiles.size).toBe(0);
      fileTree.checkboxStates.forEach(state => {
        expect(state.checked).toBe(false);
        expect(state.indeterminate).toBe(false);
      });
    });
  });

  describe('File Icon Management', () => {
    test('should return correct icon for file extensions', () => {
      const jsIcon = fileTree.getFileIcon('test.js');
      const pyIcon = fileTree.getFileIcon('script.py');
      const unknownIcon = fileTree.getFileIcon('file.unknown');
      
      expect(jsIcon).toBeDefined();
      expect(pyIcon).toBeDefined();
      expect(unknownIcon).toBeDefined();
    });

    test('should handle files without extensions', () => {
      const icon = fileTree.getFileIcon('README');
      expect(icon).toBeDefined();
    });

    test('should handle empty filename', () => {
      const icon = fileTree.getFileIcon('');
      expect(icon).toBeDefined();
    });
  });

  describe('Selected Files Management', () => {
    test('should get selected files as array', () => {
      const selected = fileTree.getSelectedFiles();
      expect(Array.isArray(selected)).toBe(true);
      expect(selected.length).toBe(0);
    });

    test('should set selected files programmatically', () => {
      const filesToSelect = ['src/index.js', 'README.md'];
      fileTree.setSelectedFiles(filesToSelect);
      
      const selected = fileTree.getSelectedFiles();
      expect(selected).toEqual(expect.arrayContaining(filesToSelect));
      
      filesToSelect.forEach(path => {
        expect(fileTree.selectedFiles.has(path)).toBe(true);
        const state = fileTree.checkboxStates.get(path);
        if (state) {
          expect(state.checked).toBe(true);
        }
      });
    });
  });

  describe('Tree Component Creation', () => {
    test('should create node component function', () => {
      const NodeComponent = fileTree.createNodeComponent();
      expect(typeof NodeComponent).toBe('function');
    });

    test('should create tree component function', () => {
      const TreeComponent = fileTree.createTreeComponent();
      expect(typeof TreeComponent).toBe('function');
    });

    test('should initialize tree component', () => {
      fileTree.initializeTreeComponent();
      expect(fileTree.FileTreeComponent).toBeDefined();
      expect(typeof fileTree.FileTreeComponent).toBe('function');
    });
  });

  describe('Tree Rendering', () => {
    test('should handle missing container element', () => {
      global.document.getElementById = () => null;
      
      const result = fileTree.renderTree('missing-container', createMockFileData());
      expect(result).toBe(false);
    });

    test('should handle rendering errors gracefully', () => {
      global.document.getElementById = () => {
        throw new Error('DOM error');
      };
      
      const result = fileTree.renderTree('container', createMockFileData());
      expect(result).toBe(false);
    });

    test('should attempt to render when container exists', () => {
      global.document.getElementById = () => ({ appendChild: () => {} });
      global.window.ReactDOM.createRoot = () => ({ render: () => {} });

      expect(() => fileTree.renderTree('container', createMockFileData())).not.toThrow();
    });
  });

  describe('Edge Cases and Error Handling', () => {
    test('should handle malformed file data', () => {
      const malformedFiles = [
        { path: 'valid/file.js' },
        { /* missing path */ },
        { path: null },
        { path: '' },
        null,
        undefined
      ];
      
      expect(() => {
        fileTree.buildTreeData(malformedFiles);
      }).not.toThrow();
    });

    test('should handle deeply nested paths', () => {
      const deepFiles = [
        { path: 'a/b/c/d/e/f/g/h/i/j/deep.js' }
      ];
      
      const treeData = fileTree.buildTreeData(deepFiles);
      expect(treeData.length).toBe(1);
      
      expect(treeData.length).toBeGreaterThan(0);
    });

    test('should handle paths with special characters', () => {
      const specialFiles = [
        { path: 'files with spaces/test.js' },
        { path: 'files-with-dashes/test.js' },
        { path: 'files_with_underscores/test.js' },
        { path: 'files.with.dots/test.js' }
      ];
      
      expect(() => {
        fileTree.buildTreeData(specialFiles);
      }).not.toThrow();
    });
  });

  describe('Performance Tests', () => {
    test('should handle large file datasets efficiently', () => {
      // Generate large dataset
      const largeFileSet = [];
      for (let i = 0; i < 1000; i++) {
        largeFileSet.push({ path: `folder${i % 10}/subfolder${i % 5}/file${i}.js` });
      }
      
      const startTime = performance.now();
      const treeData = fileTree.buildTreeData(largeFileSet);
      const endTime = performance.now();
      
      expect(treeData.length).toBeGreaterThan(0);
      expect(endTime - startTime).toBeLessThan(1000); // Should complete within 1 second
    });

    test('should efficiently manage checkbox states for large trees', () => {
      const largeFileSet = [];
      for (let i = 0; i < 500; i++) {
        largeFileSet.push({ path: `folder${i % 10}/file${i}.js` });
      }
      
      const startTime = performance.now();
      fileTree.buildTreeData(largeFileSet);
      const endTime = performance.now();
      
      expect(fileTree.checkboxStates.size).toBeGreaterThan(500);
      expect(endTime - startTime).toBeLessThan(500); // Should complete within 0.5 seconds
    });
  });
});

describe('TreeView Integration Tests', () => {
  test('should integrate with React Arborist properly', async () => {
    // Test that our component structure is compatible with React Arborist expectations
    const { default: ScribeFileTree } = await import('../index.js');
    const fileTree = new ScribeFileTree();
    
    const mockFiles = createMockFileData();
    const treeData = fileTree.buildTreeData(mockFiles);
    
    // Verify tree data structure matches React Arborist requirements
    expect(Array.isArray(treeData)).toBe(true);
    
    treeData.forEach(node => {
      expect(node.id).toBeDefined();
      expect(node.name).toBeDefined();
      expect(typeof node.isFolder).toBe('boolean');
      
      if (node.isFolder && node.children) {
        expect(Array.isArray(node.children)).toBe(true);
      }
    });
  });

  test('should maintain consistency between selected files and checkbox states', async () => {
    const { default: ScribeFileTree } = await import('../index.js');
    const fileTree = new ScribeFileTree();
    
    const mockFiles = createMockFileData();
    const treeData = fileTree.buildTreeData(mockFiles);
    
    // Build node map
    const nodeMap = new Map();
    const buildNodeMap = (nodes) => {
      nodes.forEach(node => {
        nodeMap.set(node.path, node);
        if (node.children) {
          buildNodeMap(node.children);
        }
      });
    };
    buildNodeMap(treeData);
    
    // Select some files
    fileTree.toggleFileCheckbox(nodeMap, 'src/index.js', true);
    fileTree.toggleFileCheckbox(nodeMap, 'README.md', true);
    
    // Verify consistency
    const selectedFiles = fileTree.getSelectedFiles();
    selectedFiles.forEach(path => {
      const state = fileTree.checkboxStates.get(path);
      expect(state.checked).toBe(true);
    });
  });
});
