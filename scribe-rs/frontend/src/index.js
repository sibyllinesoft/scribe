// Import React and dependencies
import React, { useState, useCallback, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Tree } from 'react-arborist';
import { 
  ChevronRight, 
  Folder, 
  FileText, 
  File,
  FolderOpen,
  Check,
  Minus 
} from 'lucide-react';

// Import CSS styles
import './checkbox-styles.css';

// Make libraries available globally for template use
window.React = React;
window.ReactDOM = { createRoot };
window.ReactArborist = { Tree };
window.LucideReact = {
  ChevronRight,
  Folder,
  FileText,
  File,
  FolderOpen,
  Check,
  Minus
};

// File Tree Implementation
class ScribeFileTree {
  constructor() {
    this.checkboxStates = new Map(); // Store checkbox states: path -> { checked: boolean, indeterminate: boolean }
    this.selectedFiles = new Set(); // Store selected file paths
    this.initializeTreeComponent();
  }

  // Create a tree node for the given path segment
  createTreeNode(part, currentPath, isLast, index, file) {
    const node = {
      id: currentPath,
      name: part,
      isFolder: !isLast,
      path: currentPath,
      fileIndex: isLast ? index : undefined,
      fileData: isLast ? file : undefined
    };

    // Initialize checkbox state for new nodes
    if (!this.checkboxStates.has(currentPath)) {
      this.checkboxStates.set(currentPath, { checked: false, indeterminate: false });
    }

    // Only add children array for folders
    if (!isLast) {
      node.children = [];
    }

    return node;
  }

  // Add a node to the tree, linking to parent if needed
  addNodeToTree(nodeMap, rootNodes, node, parentPath) {
    nodeMap.set(node.path, node);

    if (parentPath) {
      const parent = nodeMap.get(parentPath);
      if (parent && parent.children) {
        parent.children.push(node);
      }
    } else {
      rootNodes.push(node);
    }
  }

  // Process a single file path and add all path segments to the tree
  processFilePath(file, index, nodeMap, rootNodes) {
    const parts = file.path.split('/');
    let currentPath = '';

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const parentPath = currentPath;
      currentPath = currentPath ? `${currentPath}/${part}` : part;
      const isLast = i === parts.length - 1;

      if (!nodeMap.has(currentPath)) {
        const node = this.createTreeNode(part, currentPath, isLast, index, file);
        this.addNodeToTree(nodeMap, rootNodes, node, parentPath);
      }
    }
  }

  // Build hierarchical tree structure from flat file paths
  buildTreeData(files) {
    if (!files || files.length === 0) {
      console.warn('No files provided to buildTreeData');
      return [];
    }

    const nodeMap = new Map();
    const rootNodes = [];

    files.forEach((file, index) => {
      // Handle malformed file data
      if (!file || !file.path || typeof file.path !== 'string') {
        console.warn('Skipping malformed file data:', file);
        return;
      }

      this.processFilePath(file, index, nodeMap, rootNodes);
    });

    return rootNodes;
  }

  // Count checked children and check for indeterminate states
  countChildrenCheckboxStates(folder) {
    let checkedCount = 0;
    let totalCount = 0;
    let hasIndeterminate = false;

    for (const child of folder.children) {
      totalCount++;
      const childState = this.checkboxStates.get(child.path);

      if (childState.checked) {
        checkedCount++;
      } else if (childState.indeterminate) {
        hasIndeterminate = true;
      }
    }

    return { checkedCount, totalCount, hasIndeterminate };
  }

  // Calculate folder checkbox state based on children
  calculateFolderState(checkedCount, totalCount, hasIndeterminate) {
    if (checkedCount === totalCount && totalCount > 0) {
      return { checked: true, indeterminate: false };
    }
    if (checkedCount > 0 || hasIndeterminate) {
      return { checked: false, indeterminate: true };
    }
    return { checked: false, indeterminate: false };
  }

  // Checkbox management methods
  updateFolderCheckboxState(nodeMap, folderPath) {
    const folder = nodeMap.get(folderPath);
    if (!folder || !folder.children) return;

    const { checkedCount, totalCount, hasIndeterminate } = this.countChildrenCheckboxStates(folder);
    const folderState = this.checkboxStates.get(folderPath);
    const newState = this.calculateFolderState(checkedCount, totalCount, hasIndeterminate);

    folderState.checked = newState.checked;
    folderState.indeterminate = newState.indeterminate;
  }

  updateAllParentStates(nodeMap, path) {
    const parts = path.split('/');
    
    // Update parent folder states from bottom to top
    for (let i = parts.length - 2; i >= 0; i--) {
      const parentPath = parts.slice(0, i + 1).join('/');
      this.updateFolderCheckboxState(nodeMap, parentPath);
    }
  }

  toggleFileCheckbox(nodeMap, path, isFile) {
    // Handle invalid paths
    if (!path || typeof path !== 'string') {
      console.warn('Invalid path provided to toggleFileCheckbox:', path);
      return;
    }
    
    const currentState = this.checkboxStates.get(path);
    if (!currentState) {
      console.warn('No checkbox state found for path:', path);
      return;
    }
    
    const newChecked = !currentState.checked;
    
    if (isFile) {
      // Handle file checkbox toggle
      currentState.checked = newChecked;
      currentState.indeterminate = false;
      
      if (newChecked) {
        this.selectedFiles.add(path);
      } else {
        this.selectedFiles.delete(path);
      }
      
      // Update parent folder states
      this.updateAllParentStates(nodeMap, path);
    } else {
      // Handle folder checkbox toggle - apply to all children
      currentState.checked = newChecked;
      currentState.indeterminate = false;
      
      this.setChildrenCheckboxState(nodeMap, path, newChecked);
      this.updateAllParentStates(nodeMap, path);
    }
  }

  setChildrenCheckboxState(nodeMap, folderPath, checked) {
    const folder = nodeMap.get(folderPath);
    if (!folder || !folder.children) return;

    for (const child of folder.children) {
      const childState = this.checkboxStates.get(child.path);
      childState.checked = checked;
      childState.indeterminate = false;
      
      if (child.isFolder) {
        // Recursively set children if this is also a folder
        this.setChildrenCheckboxState(nodeMap, child.path, checked);
      } else if (checked) {
        this.selectedFiles.add(child.path);
      } else {
        this.selectedFiles.delete(child.path);
      }
    }
  }

  // File extensions that use FileText icon
  static CODE_EXTENSIONS = new Set([
    'js', 'jsx', 'ts', 'tsx', 'py', 'rs', 'go', 'java',
    'cpp', 'c', 'h', 'css', 'html', 'json', 'md',
    'yml', 'yaml', 'xml', 'sql', 'sh', 'bash', 'dockerfile',
    'gitignore', 'toml', 'lock'
  ]);

  // Get file icon based on extension
  getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    return ScribeFileTree.CODE_EXTENSIONS.has(ext) ? FileText : File;
  }

  // Build a node map from tree data for checkbox operations
  buildNodeMapFromTree(treeData) {
    const nodeMap = new Map();
    const addNodes = (nodes) => {
      nodes.forEach(n => {
        nodeMap.set(n.path, n);
        if (n.children) {
          addNodes(n.children);
        }
      });
    };
    addNodes(treeData);
    return nodeMap;
  }

  // Get the appropriate icon for a node's current state
  getNodeIcon(isFolder, isOpen, filename) {
    if (isFolder) {
      return isOpen ? FolderOpen : Folder;
    }
    return this.getFileIcon(filename);
  }

  // Get checkbox icon based on checkbox state
  static getCheckboxIcon(checkboxState) {
    if (checkboxState.indeterminate) return Minus;
    if (checkboxState.checked) return Check;
    return null;
  }

  // Create checkbox element for tree node
  createCheckboxElement(checkboxState, handleClick) {
    const CheckboxIcon = ScribeFileTree.getCheckboxIcon(checkboxState);
    const className = `tree-checkbox ${checkboxState.checked ? 'checked' : ''} ${checkboxState.indeterminate ? 'indeterminate' : ''}`;

    return React.createElement('div', {
      key: 'checkbox',
      className,
      onClick: handleClick
    }, CheckboxIcon ? React.createElement(CheckboxIcon, {
      className: 'checkbox-icon',
      size: 14
    }) : null);
  }

  // Create arrow element for folder nodes
  createArrowElement(isFolder, isOpen, handleClick) {
    if (!isFolder) {
      return React.createElement('div', { key: 'spacer', className: 'tree-arrow' });
    }
    return React.createElement('div', {
      key: 'arrow',
      className: `tree-arrow ${isOpen ? 'expanded' : ''}`,
      onClick: handleClick
    }, React.createElement(ChevronRight, { className: 'tree-icon', size: 16 }));
  }

  // Tree Node Component
  createNodeComponent() {
    const { useState, useCallback } = React;
    const self = this;

    return function Node({ node, style, dragHandle, tree }) {
      const isFolder = node.isFolder;
      const isOpen = tree.isOpen(node.id);
      const [, setForceUpdate] = useState(0);

      const checkboxState = self.checkboxStates.get(node.path) || { checked: false, indeterminate: false };

      const handleLabelClick = useCallback((e) => {
        e.stopPropagation();
        if (isFolder) {
          tree.toggle(node.id);
        } else if (node.fileIndex !== undefined) {
          const element = document.getElementById(`file-${node.fileIndex + 1}`);
          element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, [node.id, isFolder, tree, node.fileIndex]);

      const handleCheckboxClick = useCallback((e) => {
        e.stopPropagation();
        const nodeMap = self.buildNodeMapFromTree(tree.data);
        self.toggleFileCheckbox(nodeMap, node.path, !isFolder);
        setForceUpdate(prev => prev + 1);
      }, [node.path, isFolder, tree]);

      const IconComponent = self.getNodeIcon(isFolder, isOpen, node.name);

      return React.createElement('div', {
        ref: dragHandle,
        style: style,
        className: 'tree-node'
      }, React.createElement('div', {
        className: 'tree-node-content'
      }, [
        self.createCheckboxElement(checkboxState, handleCheckboxClick),
        self.createArrowElement(isFolder, isOpen, handleLabelClick),
        React.createElement(IconComponent, {
          key: 'icon',
          className: `tree-icon ${isFolder ? 'folder-icon' : 'file-icon'}`,
          size: 16,
          onClick: handleLabelClick
        }),
        React.createElement('span', {
          key: 'label',
          className: 'tree-label',
          title: node.path,
          onClick: handleLabelClick
        }, node.name)
      ]));
    };
  }

  // File Tree Component
  createTreeComponent() {
    const buildTreeData = this.buildTreeData.bind(this);
    const createNodeComponent = this.createNodeComponent.bind(this);
    const { useState } = React;

    return function FileTree({ fileData }) {
      const [treeData] = useState(() => {
        console.log('Building tree data from:', fileData);
        const tree = buildTreeData(fileData);
        console.log('Built tree data:', tree);
        return tree;
      });

      const NodeComponent = createNodeComponent();

      if (!treeData || treeData.length === 0) {
        return React.createElement('div', {
          style: { 
            padding: '20px', 
            textAlign: 'center', 
            color: 'var(--text-muted)' 
          }
        }, 'No files to display');
      }

      return React.createElement(Tree, {
        data: treeData,
        openByDefault: false,
        width: "100%",
        height: 400,
        padding: 25,
        rowHeight: 28,
        indent: 16,
        overscanCount: 8,
        children: NodeComponent
      });
    };
  }

  // Public method to get selected file paths
  getSelectedFiles() {
    return Array.from(this.selectedFiles);
  }

  // Public method to set selected files programmatically
  setSelectedFiles(filePaths) {
    // Clear current selection
    this.selectedFiles.clear();
    this.checkboxStates.forEach((state) => {
      state.checked = false;
      state.indeterminate = false;
    });

    // Set new selection
    filePaths.forEach(path => {
      this.selectedFiles.add(path);
      const state = this.checkboxStates.get(path);
      if (state) {
        state.checked = true;
        state.indeterminate = false;
      }
    });

    // Update parent folder states
    const nodeMap = new Map();
    // This would need to be called after tree is built
    // For now, we'll let the UI handle the updates
  }

  // Public method to clear all selections
  clearSelection() {
    this.selectedFiles.clear();
    this.checkboxStates.forEach((state) => {
      state.checked = false;
      state.indeterminate = false;
    });
  }

  initializeTreeComponent() {
    this.FileTreeComponent = this.createTreeComponent();
  }

  // Public method to render the tree
  renderTree(containerId, fileData) {
    try {
      const container = document.getElementById(containerId);
      if (container) {
        const root = createRoot(container);
        root.render(React.createElement(this.FileTreeComponent, { fileData }));
        console.log('React tree component rendered successfully');
        return true;
      } else {
        console.error(`Could not find container element: ${containerId}`);
        return false;
      }
    } catch (error) {
      console.error('Error rendering React tree:', error);
      return false;
    }
  }
}

// Make ScribeFileTree available globally
window.ScribeFileTree = ScribeFileTree;

// Export for module usage
export default ScribeFileTree;