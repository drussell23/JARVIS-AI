/**
 * TypeScript WebSocket Server Entry Point
 * Runs the unified WebSocket router with dynamic configuration
 */

import UnifiedWebSocketRouter from './UnifiedWebSocketRouter';
import { WebSocketBridge } from './bridges/WebSocketBridge';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';

// Load environment variables
dotenv.config();

// Dynamic configuration loader
interface ServerConfig {
  port: number;
  pythonBackendUrl: string;
  enableAuthentication: boolean;
  enableRateLimiting: boolean;
  enableDynamicRouting: boolean;
  routes: RouteConfig[];
}

interface RouteConfig {
  path: string;
  pattern?: string;
  capabilities: string[];
  priority: number;
  pythonModule?: string;
  handlers?: { [key: string]: string };
}

/**
 * Load configuration dynamically
 */
async function loadConfiguration(): Promise<ServerConfig> {
  // Try loading from environment variables first
  const envConfig: Partial<ServerConfig> = {
    port: parseInt(process.env.WEBSOCKET_PORT || '8001'),
    pythonBackendUrl: process.env.PYTHON_BACKEND_URL || 'http://localhost:8000',
    enableAuthentication: process.env.ENABLE_AUTH === 'true',
    enableRateLimiting: process.env.ENABLE_RATE_LIMIT !== 'false',
    enableDynamicRouting: process.env.ENABLE_DYNAMIC_ROUTING !== 'false'
  };

  // Try loading routes from config file
  const configPath = path.join(__dirname, '../../config/websocket-routes.json');
  let fileConfig: Partial<ServerConfig> = {};
  
  try {
    if (fs.existsSync(configPath)) {
      const configContent = fs.readFileSync(configPath, 'utf-8');
      fileConfig = JSON.parse(configContent);
    }
  } catch (error) {
    console.warn('Failed to load config file, using defaults:', error);
  }

  // Merge configurations
  const config: ServerConfig = {
    port: envConfig.port || fileConfig.port || 8001,
    pythonBackendUrl: envConfig.pythonBackendUrl || fileConfig.pythonBackendUrl || 'http://localhost:8000',
    enableAuthentication: envConfig.enableAuthentication ?? fileConfig.enableAuthentication ?? false,
    enableRateLimiting: envConfig.enableRateLimiting ?? fileConfig.enableRateLimiting ?? true,
    enableDynamicRouting: envConfig.enableDynamicRouting ?? fileConfig.enableDynamicRouting ?? true,
    routes: fileConfig.routes || getDefaultRoutes()
  };

  return config;
}

/**
 * Get default route configurations
 */
function getDefaultRoutes(): RouteConfig[] {
  return [
    {
      path: '/ws/vision',
      capabilities: ['vision', 'monitoring', 'analysis', 'autonomous', 'claude'],
      priority: 10,
      pythonModule: 'backend.api.unified_vision_handler',
      handlers: {
        'set_monitoring_interval': 'handleMonitoringInterval',
        'request_workspace_analysis': 'handleWorkspaceAnalysis',
        'execute_action': 'handleActionExecution',
        'vision_command': 'handleVisionCommand'
      }
    },
    {
      path: '/ws/voice',
      capabilities: ['voice', 'speech', 'commands'],
      priority: 8,
      pythonModule: 'backend.api.voice_websocket_handler'
    },
    {
      path: '/ws/automation',
      capabilities: ['automation', 'workflow', 'tasks'],
      priority: 7,
      pythonModule: 'backend.api.automation_handler'
    },
    {
      path: '/ws',
      capabilities: ['general', 'chat', 'notifications'],
      priority: 5,
      pythonModule: 'backend.api.general_handler'
    },
    {
      path: '/api/websocket/endpoints',
      capabilities: ['discovery'],
      priority: 1
    }
  ];
}

/**
 * Setup custom handlers for routes
 */
function setupCustomHandlers(router: UnifiedWebSocketRouter, config: ServerConfig): void {
  config.routes.forEach(routeConfig => {
    const route = router.registerRoute(routeConfig.path, {
      pattern: routeConfig.pattern ? new RegExp(routeConfig.pattern) : undefined,
      capabilities: routeConfig.capabilities,
      priority: routeConfig.priority,
      pythonModule: routeConfig.pythonModule
    });

    // Add middleware for logging
    route.use((message, ws, context, next) => {
      console.log(`[${new Date().toISOString()}] ${context.route} - ${message.type}`);
      next();
    });

    // Register specific handlers if provided
    if (routeConfig.handlers) {
      Object.entries(routeConfig.handlers).forEach(([messageType, handlerName]) => {
        route.on(messageType, async (msg, ws, ctx) => {
          // Dynamic handler loading
          try {
            if (ctx.pythonBridge && routeConfig.pythonModule) {
              const result = await ctx.pythonBridge.callPythonFunction(
                routeConfig.pythonModule,
                handlerName,
                [msg],
                { context: ctx.metadata }
              );
              
              if (result) {
                ws.send(JSON.stringify(result));
              }
            }
          } catch (error) {
            console.error(`Handler error for ${messageType}:`, error);
            ws.send(JSON.stringify({
              type: 'error',
              message: `Failed to handle ${messageType}`
            }));
          }
        });
      });
    }
  });

  // Add discovery endpoint handler
  router.registerRoute('/api/websocket/endpoints')
    .on('GET', async (msg, ws) => {
      ws.send(JSON.stringify({
        type: 'endpoints',
        endpoints: config.routes.map(r => ({
          path: r.path,
          capabilities: r.capabilities,
          priority: r.priority
        }))
      }));
    });
}

/**
 * Main server startup
 */
async function startServer() {
  console.log('🚀 Starting Unified WebSocket Server...');
  
  try {
    // Load configuration
    const config = await loadConfiguration();
    console.log('✅ Configuration loaded:', {
      port: config.port,
      pythonBackend: config.pythonBackendUrl,
      routes: config.routes.length
    });

    // Create router instance
    const router = new UnifiedWebSocketRouter({
      port: config.port,
      dynamicRouting: config.enableDynamicRouting,
      pythonIntegration: true,
      messageValidation: true,
      rateLimiting: config.enableRateLimiting,
      authentication: config.enableAuthentication
    });

    // Setup custom handlers
    setupCustomHandlers(router, config);

    // Event listeners
    router.on('server:start', () => {
      console.log('✅ WebSocket server started successfully');
    });

    router.on('client:connect', (clientId) => {
      console.log(`👤 Client connected: ${clientId}`);
    });

    router.on('client:disconnect', (clientId) => {
      console.log(`👤 Client disconnected: ${clientId}`);
    });

    // Start the server
    router.start();

    // Graceful shutdown
    process.on('SIGTERM', () => {
      console.log('📴 Shutting down server...');
      router.stop();
      process.exit(0);
    });

    process.on('SIGINT', () => {
      console.log('📴 Shutting down server...');
      router.stop();
      process.exit(0);
    });

  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();