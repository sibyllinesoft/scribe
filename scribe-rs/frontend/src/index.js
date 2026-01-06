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

  // Checkbox management methods
  updateFolderCheckboxState(nodeMap, folderPath) {
    const folder = nodeMap.get(folderPath);
    if (!folder || !folder.children) return;

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

    const folderState = this.checkboxStates.get(folderPath);
    if (checkedCount === totalCount && totalCount > 0) {
      folderState.checked = true;
      folderState.indeterminate = false;
    } else if (checkedCount > 0 || hasIndeterminate) {
      folderState.checked = false;
      folderState.indeterminate = true;
    } else {
      folderState.checked = false;
      folderState.indeterminate = false;
    }
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

  // Get file icon based on extension
  getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const iconMap = {
      'js': FileText, 'jsx': FileText, 'ts': FileText, 'tsx': FileText,
      'py': FileText, 'rs': FileText, 'go': FileText, 'java': FileText,
      'cpp': FileText, 'c': FileText, 'h': FileText, 'css': FileText,
      'html': FileText, 'json': FileText, 'md': FileText,
      'yml': FileText, 'yaml': FileText, 'xml': FileText, 'sql': FileText,
      'sh': FileText, 'bash': FileText, 'dockerfile': FileText,
      'gitignore': FileText, 'toml': FileText, 'lock': FileText
    };
    return iconMap[ext] || File;
  }

  // Tree Node Component
  createNodeComponent() {
    const { useState, useCallback } = React;
    const getFileIcon = this.getFileIcon;
    const checkboxStates = this.checkboxStates;
    const toggleFileCheckbox = this.toggleFileCheckbox.bind(this);

    return function Node({ node, style, dragHandle, tree }) {
      const isFolder = node.isFolder;
      const isOpen = tree.isOpen(node.id);
      const [forceUpdate, setForceUpdate] = useState(0);
      
      // Get checkbox state
      const checkboxState = checkboxStates.get(node.path) || { checked: false, indeterminate: false };
      
      const handleLabelClick = useCallback((e) => {
        e.stopPropagation();
        if (isFolder) {
          tree.toggle(node.id);
        } else {
          // Navigate to file section
          const fileIndex = node.fileIndex;
          if (fileIndex !== undefined) {
            const element = document.getElementById(`file-${fileIndex + 1}`);
            if (element) {
              element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }
        }
      }, [node.id, isFolder, tree, node.fileIndex]);

      const handleCheckboxClick = useCallback((e) => {
        e.stopPropagation();
        
        // Get the tree data to pass node map
        const treeData = tree.data;
        const nodeMap = new Map();
        
        // Rebuild node map from current tree data
        const buildNodeMap = (nodes) => {
          nodes.forEach(n => {
            nodeMap.set(n.path, n);
            if (n.children) {
              buildNodeMap(n.children);
            }
          });
        };
        buildNodeMap(treeData);
        
        toggleFileCheckbox(nodeMap, node.path, !isFolder);
        setForceUpdate(prev => prev + 1); // Force re-render
      }, [node.path, isFolder, tree, toggleFileCheckbox]);

      const IconComponent = isFolder ? (isOpen ? FolderOpen : Folder) : getFileIcon(node.name);

      // Render checkbox icon based on state
      const CheckboxIcon = checkboxState.indeterminate ? Minus : 
                          checkboxState.checked ? Check : 
                          'div'; // Empty div for unchecked state

      return React.createElement('div', {
        ref: dragHandle,
        style: style,
        className: 'tree-node'
      }, React.createElement('div', {
        className: 'tree-node-content'
      }, [
        // Checkbox
        React.createElement('div', {
          key: 'checkbox',
          className: `tree-checkbox ${checkboxState.checked ? 'checked' : ''} ${checkboxState.indeterminate ? 'indeterminate' : ''}`,
          onClick: handleCheckboxClick
        }, CheckboxIcon !== 'div' ? React.createElement(CheckboxIcon, {
          className: 'checkbox-icon',
          size: 14
        }) : null),
        
        // Arrow for folders
        isFolder && React.createElement('div', {
          key: 'arrow',
          className: `tree-arrow ${isOpen ? 'expanded' : ''}`,
          onClick: handleLabelClick
        }, React.createElement(ChevronRight, { 
          className: 'tree-icon',
          size: 16 
        })),
        !isFolder && React.createElement('div', { 
          key: 'spacer',
          className: 'tree-arrow' 
        }),
        
        // File/folder icon
        React.createElement(IconComponent, {
          key: 'icon',
          className: `tree-icon ${isFolder ? 'folder-icon' : 'file-icon'}`,
          size: 16,
          onClick: handleLabelClick
        }),
        
        // Label
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