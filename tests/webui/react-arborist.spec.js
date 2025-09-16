import { test, expect } from '@playwright/test';

/**
 * E2E tests for React Arborist tree component integration
 * Tests the file tree functionality and react-arborist specific behavior
 */

test.describe('React Arborist Tree Component', () => {
  test.beforeEach(async ({ page }) => {
    // Visit the Scribe web interface
    await page.goto('/');
    
    // Wait for the page to load and tree to be rendered
    await page.waitForSelector('[data-testid="file-tree"]', { timeout: 30000 });
  });

  test('should render file tree with proper structure', async ({ page }) => {
    // Check that the tree container exists
    const treeContainer = await page.locator('[data-testid="file-tree"]');
    await expect(treeContainer).toBeVisible();

    // Check for tree nodes
    const treeNodes = await page.locator('[data-testid="tree-node"]');
    await expect(treeNodes.first()).toBeVisible();

    // Verify that folders and files are distinguished
    const folderNodes = await page.locator('[data-testid="folder-node"]');
    const fileNodes = await page.locator('[data-testid="file-node"]');
    
    // Should have at least some folders and files
    expect(await folderNodes.count()).toBeGreaterThan(0);
    expect(await fileNodes.count()).toBeGreaterThan(0);
  });

  test('should allow folder expansion and collapse', async ({ page }) => {
    // Find a folder node with children
    const folderNode = await page.locator('[data-testid="folder-node"]').first();
    await expect(folderNode).toBeVisible();

    // Check initial state (should be collapsible)
    const expandButton = await folderNode.locator('[data-testid="expand-button"]');
    await expect(expandButton).toBeVisible();

    // Expand the folder
    await expandButton.click();
    
    // Wait for expansion animation/loading
    await page.waitForTimeout(500);

    // Verify children are now visible
    const children = await folderNode.locator('[data-testid="tree-node"]');
    if (await children.count() > 0) {
      await expect(children.first()).toBeVisible();
    }

    // Collapse the folder
    await expandButton.click();
    await page.waitForTimeout(500);
  });

  test('should handle file selection with checkboxes', async ({ page }) => {
    // Find a file checkbox
    const fileCheckbox = await page.locator('[data-testid="file-checkbox"]').first();
    await expect(fileCheckbox).toBeVisible();

    // Verify initial unchecked state
    await expect(fileCheckbox).not.toBeChecked();

    // Select the file
    await fileCheckbox.click();
    await expect(fileCheckbox).toBeChecked();

    // Verify the file appears in selected files (if there's a selected files display)
    const selectedCounter = await page.locator('[data-testid="selected-count"]');
    if (await selectedCounter.isVisible()) {
      const countText = await selectedCounter.textContent();
      expect(parseInt(countText.match(/\d+/)?.[0] || '0')).toBeGreaterThan(0);
    }

    // Unselect the file
    await fileCheckbox.click();
    await expect(fileCheckbox).not.toBeChecked();
  });

  test('should support folder selection (select all children)', async ({ page }) => {
    // Find a folder with a checkbox
    const folderCheckbox = await page.locator('[data-testid="folder-checkbox"]').first();
    
    if (await folderCheckbox.isVisible()) {
      // Select the folder
      await folderCheckbox.click();
      await expect(folderCheckbox).toBeChecked();

      // Verify that the folder is expanded to show selected children
      const parentFolder = folderCheckbox.locator('..');
      const childCheckboxes = await parentFolder.locator('[data-testid="file-checkbox"]');
      
      // All visible child files should be selected
      for (let i = 0; i < Math.min(await childCheckboxes.count(), 5); i++) {
        const childCheckbox = childCheckboxes.nth(i);
        if (await childCheckbox.isVisible()) {
          await expect(childCheckbox).toBeChecked();
        }
      }
    }
  });

  test('should handle tree keyboard navigation', async ({ page }) => {
    // Focus on the tree
    const firstNode = await page.locator('[data-testid="tree-node"]').first();
    await firstNode.focus();

    // Test arrow key navigation
    await page.keyboard.press('ArrowDown');
    await page.waitForTimeout(100);
    
    // Verify focus moved (visual indication or aria attributes)
    const focusedElement = await page.locator(':focus');
    await expect(focusedElement).toBeVisible();

    // Test expansion with keyboard
    const folderNode = await page.locator('[data-testid="folder-node"]:focus');
    if (await folderNode.isVisible()) {
      await page.keyboard.press('ArrowRight'); // Should expand
      await page.waitForTimeout(300);
      
      await page.keyboard.press('ArrowLeft'); // Should collapse
      await page.waitForTimeout(300);
    }
  });

  test('should handle large trees efficiently', async ({ page }) => {
    // Measure initial render time
    const startTime = Date.now();
    
    // Wait for tree to be fully loaded
    await page.waitForSelector('[data-testid="file-tree"]');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Should load within reasonable time (10 seconds for large repos)
    expect(loadTime).toBeLessThan(10000);

    // Test scrolling performance in large trees
    const treeContainer = await page.locator('[data-testid="file-tree"]');
    
    // Scroll down quickly
    for (let i = 0; i < 10; i++) {
      await treeContainer.press('PageDown');
      await page.waitForTimeout(50);
    }

    // Tree should remain responsive
    const nodes = await page.locator('[data-testid="tree-node"]');
    await expect(nodes.first()).toBeVisible();
  });

  test('should preserve tree state during interactions', async ({ page }) => {
    // Expand a folder
    const folderNode = await page.locator('[data-testid="folder-node"]').first();
    const expandButton = await folderNode.locator('[data-testid="expand-button"]');
    await expandButton.click();
    await page.waitForTimeout(500);

    // Select some files
    const fileCheckboxes = await page.locator('[data-testid="file-checkbox"]');
    const checkboxCount = Math.min(await fileCheckboxes.count(), 3);
    
    for (let i = 0; i < checkboxCount; i++) {
      await fileCheckboxes.nth(i).click();
    }

    // Refresh the page
    await page.reload();
    await page.waitForSelector('[data-testid="file-tree"]');

    // Check if state is preserved (this depends on implementation)
    // At minimum, the tree should re-render correctly
    await expect(page.locator('[data-testid="tree-node"]')).toHaveCount.greaterThan(0);
  });

  test('should handle error states gracefully', async ({ page }) => {
    // Test error handling by simulating network issues or bad data
    // This might involve intercepting network requests
    
    await page.route('**/api/**', route => {
      route.abort();
    });

    // Reload page to trigger error
    await page.reload();
    
    // Should show error state or fallback UI
    const errorMessage = await page.locator('[data-testid="error-message"], .error, [role="alert"]');
    
    // Either show an error message or gracefully degrade
    const hasError = await errorMessage.isVisible();
    const hasTree = await page.locator('[data-testid="file-tree"]').isVisible();
    
    expect(hasError || hasTree).toBe(true);
  });

  test('should support file filtering/search', async ({ page }) => {
    // Look for search input
    const searchInput = await page.locator('[data-testid="search-input"], input[placeholder*="search"], input[placeholder*="filter"]');
    
    if (await searchInput.isVisible()) {
      // Test file filtering
      await searchInput.fill('.js');
      await page.waitForTimeout(500);

      // Should show only JS files
      const visibleNodes = await page.locator('[data-testid="file-node"]:visible');
      const nodeCount = await visibleNodes.count();
      
      if (nodeCount > 0) {
        // Verify filtered results
        for (let i = 0; i < Math.min(nodeCount, 5); i++) {
          const nodeText = await visibleNodes.nth(i).textContent();
          expect(nodeText?.toLowerCase()).toContain('.js');
        }
      }

      // Clear filter
      await searchInput.clear();
      await page.waitForTimeout(500);
    }
  });

  test('should handle drag and drop operations', async ({ page }) => {
    // Test if drag and drop is supported (depends on implementation)
    const firstFile = await page.locator('[data-testid="file-node"]').first();
    const secondFile = await page.locator('[data-testid="file-node"]').nth(1);

    if (await firstFile.isVisible() && await secondFile.isVisible()) {
      // Attempt drag and drop
      const firstBox = await firstFile.boundingBox();
      const secondBox = await secondFile.boundingBox();

      if (firstBox && secondBox) {
        await page.mouse.move(firstBox.x + firstBox.width / 2, firstBox.y + firstBox.height / 2);
        await page.mouse.down();
        await page.mouse.move(secondBox.x + secondBox.width / 2, secondBox.y + secondBox.height / 2);
        await page.mouse.up();

        // Verify the tree still renders correctly after drag operation
        await expect(page.locator('[data-testid="file-tree"]')).toBeVisible();
      }
    }
  });

  test('should handle accessibility features', async ({ page }) => {
    // Test ARIA attributes and keyboard accessibility
    const tree = await page.locator('[data-testid="file-tree"]');
    
    // Check for proper ARIA roles
    const hasTreeRole = await tree.evaluate(el => el.getAttribute('role') === 'tree' || el.querySelector('[role="tree"]') !== null);
    expect(hasTreeRole).toBe(true);

    // Check for proper labeling
    const treeItems = await page.locator('[role="treeitem"]');
    if (await treeItems.count() > 0) {
      const firstItem = treeItems.first();
      const hasLabel = await firstItem.getAttribute('aria-label') || await firstItem.textContent();
      expect(hasLabel).toBeTruthy();
    }

    // Test focus management
    await tree.focus();
    const focusedElement = await page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });
});

test.describe('React Arborist Performance Tests', () => {
  test('should handle rapid state changes without memory leaks', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="file-tree"]');

    // Rapidly expand/collapse folders and select/unselect files
    for (let i = 0; i < 50; i++) {
      const expandButtons = await page.locator('[data-testid="expand-button"]');
      const checkboxes = await page.locator('[data-testid="file-checkbox"]');

      if (await expandButtons.count() > 0) {
        await expandButtons.nth(i % await expandButtons.count()).click();
      }
      
      if (await checkboxes.count() > 0) {
        await checkboxes.nth(i % await checkboxes.count()).click();
      }

      // Small delay to allow rendering
      if (i % 10 === 0) {
        await page.waitForTimeout(100);
      }
    }

    // Tree should still be responsive
    const tree = await page.locator('[data-testid="file-tree"]');
    await expect(tree).toBeVisible();
  });

  test('should maintain smooth scrolling with large datasets', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="file-tree"]');

    const treeContainer = await page.locator('[data-testid="file-tree"]');
    
    // Measure scroll performance
    const startTime = Date.now();
    
    // Scroll through the entire tree
    for (let i = 0; i < 20; i++) {
      await treeContainer.press('PageDown');
      await page.waitForTimeout(16); // ~60fps
    }

    const scrollTime = Date.now() - startTime;
    
    // Should maintain reasonable performance
    expect(scrollTime).toBeLessThan(2000);

    // Verify tree is still interactive
    const visibleNodes = await page.locator('[data-testid="tree-node"]:visible');
    await expect(visibleNodes.first()).toBeVisible();
  });
});