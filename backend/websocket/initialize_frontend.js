/**
 * Frontend WebSocket URL Update Script
 * Updates all frontend files to use the unified WebSocket endpoint
 */

const fs = require('fs');
const path = require('path');

const UNIFIED_WS_PORT = 8001;
const OLD_WS_URL = 'ws://localhost:8000/vision/ws/vision';
const NEW_WS_URL = `ws://localhost:${UNIFIED_WS_PORT}/ws/vision`;

const filesToUpdate = [
    '../../frontend/src/components/VisionConnection.js',
    '../../frontend/src/components/JarvisVoice.js'
];

function updateWebSocketUrls() {
    console.log('🔧 Updating frontend WebSocket URLs to use unified endpoint...\n');
    
    filesToUpdate.forEach(filePath => {
        const fullPath = path.join(__dirname, filePath);
        
        try {
            // Read file
            let content = fs.readFileSync(fullPath, 'utf8');
            
            // Count occurrences
            const occurrences = (content.match(/ws:\/\/localhost:8000\/vision\/ws\/vision/g) || []).length;
            
            if (occurrences > 0) {
                // Replace old URL with new URL
                content = content.replace(/ws:\/\/localhost:8000\/vision\/ws\/vision/g, NEW_WS_URL);
                
                // Write updated content
                fs.writeFileSync(fullPath, content, 'utf8');
                
                console.log(`✅ Updated ${path.basename(filePath)}: ${occurrences} URL(s) changed`);
            } else {
                // Check if already updated
                if (content.includes(NEW_WS_URL)) {
                    console.log(`✓ ${path.basename(filePath)} already using unified endpoint`);
                } else {
                    console.log(`⚠️  ${path.basename(filePath)} doesn't contain expected WebSocket URL`);
                }
            }
        } catch (error) {
            console.error(`❌ Error updating ${filePath}:`, error.message);
        }
    });
    
    console.log('\n📝 Creating .prettierignore to prevent URL changes...');
    
    // Create .prettierignore in frontend to prevent prettier from changing our URLs
    const prettierIgnorePath = path.join(__dirname, '../../frontend/.prettierignore');
    const prettierIgnoreContent = `# Prevent prettier from changing WebSocket URLs
src/components/VisionConnection.js
src/components/JarvisVoice.js
`;
    
    try {
        fs.writeFileSync(prettierIgnorePath, prettierIgnoreContent, 'utf8');
        console.log('✅ Created .prettierignore file');
    } catch (error) {
        console.error('❌ Error creating .prettierignore:', error.message);
    }
    
    console.log('\n✨ Frontend URL update complete!');
    console.log('\nNext steps:');
    console.log('1. Start the unified backend: cd backend && ./start_unified_backend.sh');
    console.log('2. Start the frontend: cd frontend && npm start');
    console.log('\nThe frontend will now connect to the unified WebSocket server on port 8001');
}

// Run the update
updateWebSocketUrls();