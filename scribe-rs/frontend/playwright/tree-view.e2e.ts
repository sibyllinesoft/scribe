import { test, expect } from '@playwright/test';
import path from 'path';

const SAMPLE_FILES = [
  { path: 'src/main.rs', icon: 'file', index: 0, size: '2.5 KB', tokens: '650', score: '0.85' },
  { path: 'src/lib.rs', icon: 'file', index: 1, size: '1.2 KB', tokens: '300', score: '0.75' },
  { path: 'src/web/mod.rs', icon: 'file', index: 2, size: '3.4 KB', tokens: '920', score: '0.92' },
  { path: 'README.md', icon: 'file', index: 3, size: '1.1 KB', tokens: '210', score: '0.55' }
];

test.describe('Scribe React Arborist bundle', () => {
  const bundlePath = path.resolve(__dirname, '../scribe-webservice/static/scribe-tree-bundle.js');

  test('renders bundled tree and exposes expected nodes', async ({ page }) => {
    await page.setContent('<div id="file-tree-container" style="height:400px;"></div>');
    await page.addScriptTag({ path: bundlePath });

    const renderResult = await page.evaluate((files) => {
      if (!window.ScribeFileTree) {
        throw new Error('ScribeFileTree constructor not found on window');
      }
      const tree = new window.ScribeFileTree();
      return tree.renderTree('file-tree-container', files);
    }, SAMPLE_FILES);

    expect(renderResult).toBeTruthy();

    await expect(page.locator('#file-tree-container')).toContainText('src');
    await expect(page.getByText('README.md').first()).toBeVisible();

    const treeItems = await page.locator('[role="treeitem"]').count();
    expect(treeItems).toBeGreaterThan(0);
  });
});

declare global {
  interface Window {
    ScribeFileTree?: new () => {
      renderTree: (containerId: string, fileData: typeof SAMPLE_FILES) => boolean;
    };
  }
}
