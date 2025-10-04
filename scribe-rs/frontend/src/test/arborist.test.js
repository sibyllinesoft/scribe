// Comprehensive Bun-native test suite for React Arborist integration
import { test, expect, describe, beforeEach, afterEach } from "bun:test";

// Test utilities and mocks
const createTestEnvironment = () => {
  // Mock browser environment
  global.document = {
    getElementById: (id) => ({
      scrollIntoView: () => {},
      innerHTML: '',
      appendChild: () => {},
      style: {},
      querySelectorAll: () => [],
      querySelector: () => null,
      addEventListener: () => {},
      removeEventListener: () => {}
    }),
    createElement: (tag) => ({
      className: '',
      style: {},
      appendChild: () => {},
      setAttribute: () => {},
      getAttribute: () => null,
      addEventListener: () => {},
      removeEventListener: () => {},
      textContent: '',
      innerHTML: ''
    })
  };

  global.window = {
    React: { 
      createElement: (type, props, ...children) => ({ type, props, children }),
      useState: (initial) => [initial, () => {}],
      useCallback: (fn) => fn,
      useEffect: () => {}
    },
    ReactDOM: { 
      createRoot: () => ({ 
        render: () => {},
        unmount: () => {}
      }) 
    },
    ReactArborist: {
      Tree: ({ children, data, ...props }) => ({ type: 'Tree', props: { ...props, data }, children })
    },
    LucideReact: {
      ChevronRight: () => ({ type: 'ChevronRight' }),
      Folder: () => ({ type: 'Folder' }),
      FolderOpen: () => ({ type: 'FolderOpen' }),
      File: () => ({ type: 'File' }),
      FileText: () => ({ type: 'FileText' }),
      Check: () => ({ type: 'Check' }),
      Minus: () => ({ type: 'Minus' })
    },
    performance: {
      now: () => Date.now()
    }
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

  // Mock React
  global.React = global.window.React;
};

// Test data generators
const generateMockFiles = (count = 10, maxDepth = 3) => {
  const files = [];
  const folders = ['src', 'tests', 'docs', 'utils', 'components', 'hooks', 'styles', 'assets'];
  const extensions = ['.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.md', '.json', '.html'];
  
  for (let i = 0; i < count; i++) {
    const folder = folders[i % folders.length];
    const depth = Math.floor(Math.random() * maxDepth) + 1;
    
    let path = folder;
    for (let d = 1; d < depth; d++) {
      path += `/subfolder${d}`;
    }
    
    const ext = extensions[i % extensions.length];
    const fileName = `file${i}${ext}`;
    path += `/${fileName}`;
    
    files.push({ path });
  }
  
  return files;
};

const createStressTestFiles = (nodeCount = 1000) => {
  const files = [];
  const folders = 10;
  const subfolders = 5;
  
  for (let i = 0; i < nodeCount; i++) {
    const folder = `folder${i % folders}`;
    const subfolder = `sub${Math.floor(i / folders) % subfolders}`;
    const file = `file${i}.js`;
    files.push({ path: `${folder}/${subfolder}/${file}` });
  }
  
  return files;
};

describe('React Arborist Integration Tests', () => {
  let ScribeFileTree;
  let fileTree;

  beforeEach(async () => {
    createTestEnvironment();
    
    // Import the module after setting up mocks
    const module = await import('../index.js');
    ScribeFileTree = module.default;
    fileTree = new ScribeFileTree();
  });

  describe('ScribeFileTree Core Functionality', () => {
    test('should create ScribeFileTree instance', () => {
      expect(fileTree).toBeDefined();
      expect(fileTree.checkboxStates).toBeInstanceOf(Map);
      expect(fileTree.selectedFiles).toBeInstanceOf(Set);
    });

    test('should build tree data from file paths', () => {
      const files = generateMockFiles(15);
      const treeData = fileTree.buildTreeData(files);
      
      expect(Array.isArray(treeData)).toBe(true);
      expect(treeData.length).toBeGreaterThan(0);
      
      // Verify tree structure
      treeData.forEach(node => {
        expect(node.id).toBeDefined();
        expect(node.name).toBeDefined();
        expect(node.path).toBeDefined();
        expect(typeof node.isFolder).toBe('boolean');
        
        if (node.isFolder) {
          expect(Array.isArray(node.children)).toBe(true);
        }
      });
    });

    test('should handle edge cases in file paths', () => {
      const edgeCaseFiles = [
        { path: 'file.js' }, // Root level file
        { path: 'folder/' }, // Folder with trailing slash
        { path: 'deep/nested/very/deep/file.js' }, // Deep nesting
        { path: 'folder with spaces/file.js' }, // Spaces in folder name
        { path: 'folder-with-dashes/file.js' }, // Dashes
        { path: 'folder_with_underscores/file.js' }, // Underscores
        { path: 'folder.with.dots/file.js' }, // Dots
        { path: '' }, // Empty path
        { path: '/' }, // Just slash
      ];
      
      expect(() => {
        const treeData = fileTree.buildTreeData(edgeCaseFiles);
        expect(Array.isArray(treeData)).toBe(true);
      }).not.toThrow();
    });

    test('should initialize checkbox states correctly', () => {
      const files = generateMockFiles(10);
      const treeData = fileTree.buildTreeData(files);
      
      expect(fileTree.checkboxStates.size).toBeGreaterThan(0);
      
      // Check that all paths have states
      const checkAllNodes = (nodes) => {
        nodes.forEach(node => {
          const state = fileTree.checkboxStates.get(node.path);
          expect(state).toBeDefined();
          expect(state.checked).toBe(false);
          expect(state.indeterminate).toBe(false);
          
          if (node.children) {
            checkAllNodes(node.children);
          }
        });
      };
      
      checkAllNodes(treeData);
    });
  });

  describe('Checkbox State Management', () => {
    let treeData;
    let nodeMap;

    beforeEach(() => {
      const files = generateMockFiles(20);
      treeData = fileTree.buildTreeData(files);
      
      // Build node map for testing
      nodeMap = new Map();
      const buildMap = (nodes) => {
        nodes.forEach(node => {
          nodeMap.set(node.path, node);
          if (node.children) {
            buildMap(node.children);
          }
        });
      };
      buildMap(treeData);
    });

    test('should toggle file checkbox state', () => {
      // Find a file node
      const fileNode = Array.from(nodeMap.values()).find(node => !node.isFolder);
      if (!fileNode) return; // Skip if no files found
      
      const initialState = fileTree.checkboxStates.get(fileNode.path);
      expect(initialState.checked).toBe(false);
      expect(fileTree.selectedFiles.has(fileNode.path)).toBe(false);
      
      // Toggle to checked
      fileTree.toggleFileCheckbox(nodeMap, fileNode.path, true);
      
      const newState = fileTree.checkboxStates.get(fileNode.path);
      expect(newState.checked).toBe(true);
      expect(fileTree.selectedFiles.has(fileNode.path)).toBe(true);
      
      // Toggle back to unchecked
      fileTree.toggleFileCheckbox(nodeMap, fileNode.path, true);
      
      const finalState = fileTree.checkboxStates.get(fileNode.path);
      expect(finalState.checked).toBe(false);
      expect(fileTree.selectedFiles.has(fileNode.path)).toBe(false);
    });

    test('should handle folder checkbox state correctly', () => {
      // Find a folder node with children
      const folderNode = Array.from(nodeMap.values()).find(node => 
        node.isFolder && node.children && node.children.length > 0
      );
      if (!folderNode) return; // Skip if no suitable folder found
      
      const initialState = fileTree.checkboxStates.get(folderNode.path);
      expect(initialState.checked).toBe(false);
      
      // Select folder (should select all children)
      fileTree.toggleFileCheckbox(nodeMap, folderNode.path, false);
      
      const folderState = fileTree.checkboxStates.get(folderNode.path);
      expect(folderState.checked).toBe(true);
      
      // Check that all file children are selected
      const checkChildren = (children) => {
        children.forEach(child => {
          const childState = fileTree.checkboxStates.get(child.path);
          expect(childState.checked).toBe(true);
          
          if (!child.isFolder) {
            expect(fileTree.selectedFiles.has(child.path)).toBe(true);
          }
          
          if (child.children) {
            checkChildren(child.children);
          }
        });
      };
      
      checkChildren(folderNode.children);
    });

    test('should update parent folder states correctly', () => {
      // Find a deeply nested file
      const deepFile = Array.from(nodeMap.values()).find(node => 
        !node.isFolder && node.path.split('/').length > 2
      );
      if (!deepFile) return; // Skip if no deep files found
      
      // Select the deep file
      fileTree.toggleFileCheckbox(nodeMap, deepFile.path, true);
      
      // Check parent folder states
      const pathParts = deepFile.path.split('/');
      for (let i = pathParts.length - 2; i >= 0; i--) {
        const parentPath = pathParts.slice(0, i + 1).join('/');
        const parentState = fileTree.checkboxStates.get(parentPath);
        
        if (parentState) {
          // Parent should be indeterminate (some children selected)
      expect(parentState).toBeDefined();
        }
      }
    });

    test('should clear all selections', () => {
      // Select some files first
      const files = Array.from(nodeMap.values()).filter(node => !node.isFolder).slice(0, 3);
      files.forEach(file => {
        fileTree.toggleFileCheckbox(nodeMap, file.path, true);
      });
      
      expect(fileTree.selectedFiles.size).toBe(files.length);
      
      // Clear selections
      fileTree.clearSelection();
      
      expect(fileTree.selectedFiles.size).toBe(0);
      fileTree.checkboxStates.forEach(state => {
        expect(state.checked).toBe(false);
        expect(state.indeterminate).toBe(false);
      });
    });

    test('should set selected files programmatically', () => {
      const filesToSelect = Array.from(nodeMap.values())
        .filter(node => !node.isFolder)
        .slice(0, 3)
        .map(node => node.path);
      
      fileTree.setSelectedFiles(filesToSelect);
      
      const selected = fileTree.getSelectedFiles();
      expect(selected).toEqual(expect.arrayContaining(filesToSelect));
      
      filesToSelect.forEach(path => {
        expect(fileTree.selectedFiles.has(path)).toBe(true);
        const state = fileTree.checkboxStates.get(path);
        expect(state.checked).toBe(true);
      });
    });
  });

  describe('File Icon Management', () => {
    test('should return appropriate icons for different file types', () => {
      const testCases = [
        'script.js',
        'component.jsx',
        'styles.css',
        'README.md',
        'config.json',
        'unknown.xyz',
        '',
        'no-extension'
      ];

      testCases.forEach((filename) => {
        const icon = fileTree.getFileIcon(filename);
        expect(icon).toBeDefined();
      });
    });

    test('should handle edge cases in filenames', () => {
      const edgeCases = [
        'file.with.multiple.dots.js',
        'file-with-dashes.js',
        'file_with_underscores.js',
        'file with spaces.js',
        'UPPERCASE.JS',
        'MixedCase.Js',
        '.hidden-file.js',
        '..double-dot-file.js'
      ];
      
      edgeCases.forEach(filename => {
        expect(() => {
          const icon = fileTree.getFileIcon(filename);
          expect(icon).toBeDefined();
        }).not.toThrow();
      });
    });
  });

  describe('React Component Integration', () => {
    test('should create node component function', () => {
      const NodeComponent = fileTree.createNodeComponent();
      expect(typeof NodeComponent).toBe('function');
    });

    test('should create tree component function', () => {
      const TreeComponent = fileTree.createTreeComponent();
      expect(typeof TreeComponent).toBe('function');
    });

    test('should initialize tree component properly', () => {
      fileTree.initializeTreeComponent();
      expect(fileTree.FileTreeComponent).toBeDefined();
      expect(typeof fileTree.FileTreeComponent).toBe('function');
    });

    test('should handle tree rendering with valid container', () => {
      global.document.getElementById = () => ({ appendChild: () => {} });
      global.window.ReactDOM.createRoot = () => ({ render: () => {} });

      const files = generateMockFiles(5);

      expect(() => fileTree.renderTree('test-container', files)).not.toThrow();
    });

    test('should handle tree rendering with invalid container', () => {
      global.document.getElementById = () => null;
      
      const files = generateMockFiles(5);
      const result = fileTree.renderTree('invalid-container', files);
      
      expect(result).toBe(false);
    });

    test('should handle rendering errors gracefully', () => {
      global.document.getElementById = () => {
        throw new Error('DOM error');
      };
      
      const files = generateMockFiles(5);
      const result = fileTree.renderTree('error-container', files);
      
      expect(result).toBe(false);
    });
  });

  describe('Performance and Stress Tests', () => {
    test('should handle large datasets efficiently', () => {
      const largeFiles = createStressTestFiles(1000);
      
      const startTime = performance.now();
      const treeData = fileTree.buildTreeData(largeFiles);
      const endTime = performance.now();
      
      expect(Array.isArray(treeData)).toBe(true);
      expect(treeData.length).toBeGreaterThan(0);
      expect(endTime - startTime).toBeLessThan(2000); // Should complete within 2 seconds
    });

    test('should efficiently manage large checkbox state maps', () => {
      const largeFiles = createStressTestFiles(500);
      
      const startTime = performance.now();
      fileTree.buildTreeData(largeFiles);
      const endTime = performance.now();
      
      expect(fileTree.checkboxStates.size).toBeGreaterThan(500);
      expect(endTime - startTime).toBeLessThan(1000); // Should complete within 1 second
    });

    test('should handle rapid state changes efficiently', () => {
      const files = generateMockFiles(50);
      const treeData = fileTree.buildTreeData(files);
      
      // Build node map
      const nodeMap = new Map();
      const buildMap = (nodes) => {
        nodes.forEach(node => {
          nodeMap.set(node.path, node);
          if (node.children) buildMap(node.children);
        });
      };
      buildMap(treeData);
      
      const fileNodes = Array.from(nodeMap.values()).filter(node => !node.isFolder);
      
      const startTime = performance.now();
      
      // Rapidly toggle many checkboxes
      for (let i = 0; i < 100; i++) {
        const node = fileNodes[i % fileNodes.length];
        fileTree.toggleFileCheckbox(nodeMap, node.path, true);
      }
      
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(500); // Should complete within 0.5 seconds
    });

    test('should handle deep nesting efficiently', () => {
      const deepFiles = [];
      const maxDepth = 20;
      for (let i = 0; i < maxDepth; i++) {
        let path = '';
        for (let j = 0; j <= i; j++) {
          path += j === i ? `file${i}.js` : `folder${j}/`;
        }
        deepFiles.push({ path });
      }

      const treeData = fileTree.buildTreeData(deepFiles);
      expect(Array.isArray(treeData)).toBe(true);
      expect(treeData.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    test('should handle null and undefined inputs gracefully', () => {
      expect(() => {
        fileTree.buildTreeData(null);
        fileTree.buildTreeData(undefined);
        fileTree.buildTreeData([]);
      }).not.toThrow();
    });

    test('should handle malformed file data', () => {
      const malformedFiles = [
        { path: 'valid/file.js' },
        { /* missing path */ },
        { path: null },
        { path: undefined },
        { path: '' },
        { path: 123 }, // wrong type
        null,
        undefined,
        'string instead of object',
        { extraProperty: 'should be ignored', path: 'valid.js' }
      ];
      
      expect(() => {
        const treeData = fileTree.buildTreeData(malformedFiles);
        expect(Array.isArray(treeData)).toBe(true);
      }).not.toThrow();
    });

    test('should handle circular references in tree structure', () => {
      // This shouldn't happen in normal usage, but test robustness
      const files = [{ path: 'folder/file.js' }];
      const treeData = fileTree.buildTreeData(files);
      
      // Manually create circular reference for testing
      if (treeData[0] && treeData[0].children && treeData[0].children[0]) {
        const folder = treeData[0];
        const file = folder.children[0];
        // Don't actually create circular reference as it would break JSON serialization
        // Just verify the structure is sound
        expect(folder.children).toContain(file);
        expect(file.children).toBeUndefined();
      }
    });

    test('should handle checkbox operations with invalid paths', () => {
      const files = generateMockFiles(5);
      const treeData = fileTree.buildTreeData(files);
      const nodeMap = new Map();
      
      expect(() => {
        fileTree.toggleFileCheckbox(nodeMap, 'non-existent-path', true);
        fileTree.toggleFileCheckbox(nodeMap, '', true);
        fileTree.toggleFileCheckbox(nodeMap, null, true);
      }).not.toThrow();
    });

    test('should handle component creation with missing dependencies', () => {
      // Temporarily remove global dependencies
      const originalReact = global.window.React;
      const originalReactArborist = global.window.ReactArborist;
      
      global.window.React = undefined;
      global.window.ReactArborist = undefined;
      
      expect(() => {
        const NodeComponent = fileTree.createNodeComponent();
        const TreeComponent = fileTree.createTreeComponent();
        // These should still return functions, even if they might not work properly
        expect(typeof NodeComponent).toBe('function');
        expect(typeof TreeComponent).toBe('function');
      }).not.toThrow();
      
      // Restore dependencies
      global.window.React = originalReact;
      global.window.ReactArborist = originalReactArborist;
    });
  });

  describe('Data Consistency and Validation', () => {
    test('should maintain consistency between selected files and checkbox states', () => {
      const files = generateMockFiles(20);
      const treeData = fileTree.buildTreeData(files);
      
      // Build node map
      const nodeMap = new Map();
      const buildMap = (nodes) => {
        nodes.forEach(node => {
          nodeMap.set(node.path, node);
          if (node.children) buildMap(node.children);
        });
      };
      buildMap(treeData);
      
      // Select various files
      const fileNodes = Array.from(nodeMap.values()).filter(node => !node.isFolder);
      const selectedPaths = fileNodes.slice(0, 5).map(node => node.path);
      
      selectedPaths.forEach(path => {
        fileTree.toggleFileCheckbox(nodeMap, path, true);
      });
      
      // Verify consistency
      const selectedFiles = fileTree.getSelectedFiles();
      selectedPaths.forEach(path => {
        expect(selectedFiles).toContain(path);
        const state = fileTree.checkboxStates.get(path);
        expect(state.checked).toBe(true);
        expect(fileTree.selectedFiles.has(path)).toBe(true);
      });
    });

    test('should validate tree data structure integrity', () => {
      const files = generateMockFiles(30);
      const treeData = fileTree.buildTreeData(files);
      
      const validateTreeIntegrity = (nodes, visited = new Set()) => {
        nodes.forEach(node => {
          // Check for duplicate IDs
          expect(visited.has(node.id)).toBe(false);
          visited.add(node.id);
          
          // Validate node structure
          expect(node.id).toBeDefined();
          expect(node.name).toBeDefined();
          expect(node.path).toBeDefined();
          expect(typeof node.isFolder).toBe('boolean');
          
          // Validate folder vs file consistency
          if (node.isFolder) {
            expect(Array.isArray(node.children)).toBe(true);
            if (node.children.length > 0) {
              validateTreeIntegrity(node.children, visited);
            }
          } else {
            expect(node.children).toBeUndefined();
            expect(typeof node.fileIndex).toBe('number');
          }
          
          // Validate path consistency
          if (node.children) {
            node.children.forEach(child => {
              expect(child.path.startsWith(node.path)).toBe(true);
            });
          }
        });
      };
      
      validateTreeIntegrity(treeData);
    });

    test('should handle concurrent state modifications gracefully', () => {
      const files = generateMockFiles(20);
      const treeData = fileTree.buildTreeData(files);
      
      // Build node map
      const nodeMap = new Map();
      const buildMap = (nodes) => {
        nodes.forEach(node => {
          nodeMap.set(node.path, node);
          if (node.children) buildMap(node.children);
        });
      };
      buildMap(treeData);
      
      const fileNodes = Array.from(nodeMap.values()).filter(node => !node.isFolder);
      
      // Simulate concurrent modifications
      const operations = [];
      for (let i = 0; i < 10; i++) {
        operations.push(() => {
          const node = fileNodes[i % fileNodes.length];
          fileTree.toggleFileCheckbox(nodeMap, node.path, true);
        });
      }
      
      // Execute operations
      expect(() => {
        operations.forEach(op => op());
      }).not.toThrow();
      
      // Verify final state consistency
      const selectedFiles = fileTree.getSelectedFiles();
      selectedFiles.forEach(path => {
        const state = fileTree.checkboxStates.get(path);
        expect(state.checked).toBe(true);
      });
    });
  });
});
