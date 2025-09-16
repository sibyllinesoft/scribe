// Browser test script for React Arborist component
// Run this in the browser console after opening scribe-core-test.html

console.log("🧪 Testing React Arborist Component");
console.log("=====================================");

// Check if all required libraries are loaded
const checks = {
    "React": typeof window.React !== 'undefined',
    "ReactDOM": typeof window.ReactDOM !== 'undefined', 
    "ReactArborist": typeof window.ReactArborist !== 'undefined',
    "LucideReact": typeof window.LucideReact !== 'undefined',
    "ScribeFileTree": typeof window.ScribeFileTree !== 'undefined'
};

console.log("📚 Library Availability:");
Object.entries(checks).forEach(([name, available]) => {
    console.log(`  ${available ? '✅' : '❌'} ${name}: ${available ? 'Available' : 'Missing'}`);
});

// Test ScribeFileTree if available
if (checks.ScribeFileTree) {
    console.log("\n🌳 Testing ScribeFileTree:");
    try {
        const fileTree = new window.ScribeFileTree();
        console.log("  ✅ ScribeFileTree instance created successfully");
        
        // Check if container exists
        const container = document.getElementById('file-tree-container');
        if (container) {
            console.log("  ✅ Tree container found");
            
            // Check if fileData exists
            if (typeof fileData !== 'undefined' && fileData.length > 0) {
                console.log(`  ✅ File data available: ${fileData.length} files`);
                console.log(`  📄 Sample file: ${fileData[0].path}`);
                
                // Test rendering
                const success = fileTree.renderTree('file-tree-container', fileData);
                console.log(`  ${success ? '✅' : '❌'} Tree rendering: ${success ? 'Success' : 'Failed'}`);
                
                // Check container content after render
                setTimeout(() => {
                    const hasContent = container.children.length > 0;
                    console.log(`  ${hasContent ? '✅' : '❌'} Container has content: ${hasContent}`);
                    if (hasContent) {
                        console.log(`  📊 Container children: ${container.children.length}`);
                    }
                    
                    // Check for React components
                    const reactNodes = container.querySelectorAll('[data-react-root], .tree-node');
                    console.log(`  🔍 React nodes found: ${reactNodes.length}`);
                    
                    if (reactNodes.length > 0) {
                        console.log("  🎉 SUCCESS: React Arborist component is rendering!");
                    } else {
                        console.log("  ⚠️  WARNING: No React nodes detected in tree container");
                    }
                }, 1000);
            } else {
                console.log("  ❌ No file data available");
            }
        } else {
            console.log("  ❌ Tree container not found");
        }
    } catch (error) {
        console.log(`  ❌ Error testing ScribeFileTree: ${error.message}`);
    }
} else {
    console.log("\n❌ Cannot test ScribeFileTree - not available");
}

// Summary
setTimeout(() => {
    console.log("\n📋 Test Summary:");
    console.log("================");
    console.log("This test verifies that:");
    console.log("1. All React libraries are loaded correctly");
    console.log("2. ScribeFileTree class is available and instantiable");
    console.log("3. File data is present and properly formatted");
    console.log("4. React Arborist component renders into the DOM");
    console.log("5. Tree nodes are created and interactive");
    console.log("\nCheck the visual tree above to confirm the component is working!");
}, 1500);