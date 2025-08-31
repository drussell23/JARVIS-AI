import { UnifiedWebSocketRouter } from './UnifiedWebSocketRouter';
import * as path from 'path';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const port = parseInt(process.env.PORT || '8001', 10);

console.log('🚀 Starting Unified WebSocket Server...');

// Create and configure the router
const router = new UnifiedWebSocketRouter({ port });

// Start the server
router.start();

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('📛 SIGTERM signal received: closing WebSocket server');
  router.stop();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('📛 SIGINT signal received: closing WebSocket server');
  router.stop();
  process.exit(0);
});