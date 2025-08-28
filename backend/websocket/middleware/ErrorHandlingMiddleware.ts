/**
 * Advanced Error Handling and Reconnection Middleware
 * Provides comprehensive error recovery, circuit breaking, and intelligent reconnection
 */

import { WebSocket } from 'ws';
import { EventEmitter } from 'events';

interface ErrorContext {
  route: string;
  clientId: string;
  messageType: string;
  timestamp: Date;
  error: Error;
  attemptNumber: number;
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeout: number;
  halfOpenRequests: number;
}

interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
}

enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

export class ErrorHandlingMiddleware extends EventEmitter {
  private errorCounts: Map<string, number> = new Map();
  private errorHistory: ErrorContext[] = [];
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private retryQueues: Map<string, RetryQueue> = new Map();
  
  constructor(
    private config: {
      circuitBreaker?: CircuitBreakerConfig;
      retry?: RetryConfig;
      maxErrorHistory?: number;
      errorThreshold?: number;
      errorWindow?: number;
    } = {}
  ) {
    super();
    
    // Default configuration
    this.config = {
      circuitBreaker: {
        failureThreshold: 5,
        resetTimeout: 60000,
        halfOpenRequests: 3,
        ...config.circuitBreaker
      },
      retry: {
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 30000,
        backoffMultiplier: 2,
        ...config.retry
      },
      maxErrorHistory: 1000,
      errorThreshold: 10,
      errorWindow: 60000,
      ...config
    };
  }
  
  /**
   * Error handling middleware function
   */
  async handleError(
    error: Error,
    message: any,
    ws: WebSocket,
    context: any
  ): Promise<void> {
    const errorContext: ErrorContext = {
      route: context.route,
      clientId: context.clientId,
      messageType: message?.type || 'unknown',
      timestamp: new Date(),
      error,
      attemptNumber: context.attemptNumber || 1
    };
    
    // Log error
    this.logError(errorContext);
    
    // Update error metrics
    this.updateErrorMetrics(errorContext);
    
    // Check circuit breaker
    const circuitBreaker = this.getCircuitBreaker(context.route);
    if (circuitBreaker.state === CircuitState.OPEN) {
      this.sendErrorResponse(ws, {
        type: 'error',
        code: 'CIRCUIT_OPEN',
        message: 'Service temporarily unavailable',
        retryAfter: circuitBreaker.nextAttempt
      });
      return;
    }
    
    // Record failure in circuit breaker
    circuitBreaker.recordFailure();
    
    // Determine if we should retry
    if (this.shouldRetry(error, errorContext)) {
      await this.scheduleRetry(message, ws, context);
    } else {
      // Send final error response
      this.sendErrorResponse(ws, {
        type: 'error',
        code: this.getErrorCode(error),
        message: this.sanitizeErrorMessage(error),
        details: this.getErrorDetails(error, errorContext)
      });
    }
    
    // Check if we need to alert
    if (this.shouldAlert(errorContext)) {
      this.emit('alert', {
        severity: 'high',
        error: errorContext,
        message: 'High error rate detected'
      });
    }
  }
  
  /**
   * Recovery middleware for successful operations
   */
  async handleSuccess(
    message: any,
    ws: WebSocket,
    context: any
  ): Promise<void> {
    const circuitBreaker = this.getCircuitBreaker(context.route);
    circuitBreaker.recordSuccess();
    
    // Clear retry queue for this client
    const retryKey = `${context.clientId}:${message.type}`;
    this.retryQueues.delete(retryKey);
  }
  
  /**
   * Reconnection handler
   */
  async handleReconnection(
    ws: WebSocket,
    clientId: string,
    lastMessageId?: string
  ): Promise<void> {
    // Send reconnection acknowledgment
    ws.send(JSON.stringify({
      type: 'reconnected',
      clientId,
      timestamp: new Date().toISOString(),
      lastMessageId
    }));
    
    // Replay any queued messages
    const clientQueue = this.getClientRetryQueue(clientId);
    if (clientQueue.length > 0) {
      for (const queuedMessage of clientQueue) {
        ws.send(JSON.stringify(queuedMessage));
      }
      
      // Clear the queue
      this.clearClientRetryQueue(clientId);
    }
    
    this.emit('client:reconnected', { clientId, queuedMessages: clientQueue.length });
  }
  
  /**
   * Get or create circuit breaker for a route
   */
  private getCircuitBreaker(route: string): CircuitBreaker {
    if (!this.circuitBreakers.has(route)) {
      this.circuitBreakers.set(
        route,
        new CircuitBreaker(route, this.config.circuitBreaker!)
      );
    }
    return this.circuitBreakers.get(route)!;
  }
  
  /**
   * Determine if we should retry the operation
   */
  private shouldRetry(error: Error, context: ErrorContext): boolean {
    // Don't retry if max retries exceeded
    if (context.attemptNumber >= this.config.retry!.maxRetries) {
      return false;
    }
    
    // Check if error is retryable
    return this.isRetryableError(error);
  }
  
  /**
   * Check if an error is retryable
   */
  private isRetryableError(error: Error): boolean {
    // Network errors
    if (error.message.includes('ECONNRESET') ||
        error.message.includes('ETIMEDOUT') ||
        error.message.includes('ENOTFOUND')) {
      return true;
    }
    
    // Temporary failures
    if (error.message.includes('temporarily unavailable') ||
        error.message.includes('rate limit')) {
      return true;
    }
    
    // Don't retry on client errors
    if (error.message.includes('Invalid') ||
        error.message.includes('Unauthorized')) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Schedule a retry
   */
  private async scheduleRetry(
    message: any,
    ws: WebSocket,
    context: any
  ): Promise<void> {
    const delay = this.calculateRetryDelay(context.attemptNumber);
    const retryKey = `${context.clientId}:${message.type}`;
    
    // Add to retry queue
    if (!this.retryQueues.has(retryKey)) {
      this.retryQueues.set(retryKey, new RetryQueue());
    }
    
    const retryQueue = this.retryQueues.get(retryKey)!;
    retryQueue.add({
      message,
      context: { ...context, attemptNumber: (context.attemptNumber || 0) + 1 },
      scheduledAt: new Date(),
      executeAt: new Date(Date.now() + delay)
    });
    
    // Send retry notification
    ws.send(JSON.stringify({
      type: 'retry_scheduled',
      originalMessageId: message.id,
      retryIn: delay,
      attempt: context.attemptNumber + 1
    }));
    
    // Schedule the retry
    setTimeout(() => {
      this.emit('retry', { message, ws, context });
    }, delay);
  }
  
  /**
   * Calculate retry delay with exponential backoff
   */
  private calculateRetryDelay(attemptNumber: number): number {
    const { baseDelay, maxDelay, backoffMultiplier } = this.config.retry!;
    const delay = Math.min(
      baseDelay * Math.pow(backoffMultiplier, attemptNumber - 1),
      maxDelay
    );
    
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.1 * delay;
    return Math.floor(delay + jitter);
  }
  
  /**
   * Log error with context
   */
  private logError(context: ErrorContext): void {
    console.error(`[${context.route}] Error in ${context.messageType}:`, {
      clientId: context.clientId,
      error: context.error.message,
      stack: context.error.stack,
      attempt: context.attemptNumber
    });
    
    // Add to history
    this.errorHistory.push(context);
    
    // Trim history if needed
    if (this.errorHistory.length > this.config.maxErrorHistory!) {
      this.errorHistory.shift();
    }
  }
  
  /**
   * Update error metrics
   */
  private updateErrorMetrics(context: ErrorContext): void {
    const key = `${context.route}:${context.messageType}`;
    this.errorCounts.set(key, (this.errorCounts.get(key) || 0) + 1);
  }
  
  /**
   * Check if we should send an alert
   */
  private shouldAlert(context: ErrorContext): boolean {
    const recentErrors = this.errorHistory.filter(
      e => Date.now() - e.timestamp.getTime() < this.config.errorWindow!
    );
    
    return recentErrors.length >= this.config.errorThreshold!;
  }
  
  /**
   * Get error code from error
   */
  private getErrorCode(error: Error): string {
    if (error.message.includes('timeout')) return 'TIMEOUT';
    if (error.message.includes('Invalid')) return 'INVALID_REQUEST';
    if (error.message.includes('Unauthorized')) return 'UNAUTHORIZED';
    if (error.message.includes('not found')) return 'NOT_FOUND';
    return 'INTERNAL_ERROR';
  }
  
  /**
   * Sanitize error message for client
   */
  private sanitizeErrorMessage(error: Error): string {
    // Remove sensitive information
    let message = error.message;
    message = message.replace(/\/([^\/\s]+)\/([^\/\s]+)/g, '/<path>/<path>');
    message = message.replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, '<ip>');
    return message;
  }
  
  /**
   * Get error details for debugging
   */
  private getErrorDetails(error: Error, context: ErrorContext): any {
    if (process.env.NODE_ENV === 'production') {
      return undefined;
    }
    
    return {
      stack: error.stack,
      route: context.route,
      messageType: context.messageType,
      attemptNumber: context.attemptNumber
    };
  }
  
  /**
   * Send error response to client
   */
  private sendErrorResponse(ws: WebSocket, response: any): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(response));
    }
  }
  
  /**
   * Get retry queue for a client
   */
  private getClientRetryQueue(clientId: string): any[] {
    const messages: any[] = [];
    
    this.retryQueues.forEach((queue, key) => {
      if (key.startsWith(`${clientId}:`)) {
        messages.push(...queue.getAll());
      }
    });
    
    return messages;
  }
  
  /**
   * Clear retry queue for a client
   */
  private clearClientRetryQueue(clientId: string): void {
    const keysToDelete: string[] = [];
    
    this.retryQueues.forEach((_, key) => {
      if (key.startsWith(`${clientId}:`)) {
        keysToDelete.push(key);
      }
    });
    
    keysToDelete.forEach(key => this.retryQueues.delete(key));
  }
  
  /**
   * Get error statistics
   */
  getStats(): any {
    const stats = {
      totalErrors: this.errorHistory.length,
      recentErrors: this.errorHistory.filter(
        e => Date.now() - e.timestamp.getTime() < 300000
      ).length,
      errorCounts: Object.fromEntries(this.errorCounts),
      circuitBreakers: Array.from(this.circuitBreakers.entries()).map(
        ([route, cb]) => ({
          route,
          state: cb.state,
          failures: cb.failures,
          successes: cb.successes
        })
      ),
      retryQueues: Array.from(this.retryQueues.entries()).map(
        ([key, queue]) => ({
          key,
          size: queue.size()
        })
      )
    };
    
    return stats;
  }
}

/**
 * Circuit breaker implementation
 */
class CircuitBreaker {
  state: CircuitState = CircuitState.CLOSED;
  failures: number = 0;
  successes: number = 0;
  nextAttempt: Date = new Date();
  private halfOpenRequests: number = 0;
  
  constructor(
    private route: string,
    private config: CircuitBreakerConfig
  ) {}
  
  recordSuccess(): void {
    this.failures = 0;
    
    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      if (this.successes >= this.config.halfOpenRequests) {
        this.state = CircuitState.CLOSED;
        this.halfOpenRequests = 0;
      }
    }
  }
  
  recordFailure(): void {
    this.failures++;
    this.successes = 0;
    
    if (this.failures >= this.config.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.nextAttempt = new Date(Date.now() + this.config.resetTimeout);
      
      // Schedule transition to half-open
      setTimeout(() => {
        this.state = CircuitState.HALF_OPEN;
        this.halfOpenRequests = 0;
      }, this.config.resetTimeout);
    }
  }
}

/**
 * Retry queue implementation
 */
class RetryQueue {
  private queue: any[] = [];
  
  add(item: any): void {
    this.queue.push(item);
  }
  
  getAll(): any[] {
    return [...this.queue];
  }
  
  size(): number {
    return this.queue.length;
  }
  
  clear(): void {
    this.queue = [];
  }
}

export default ErrorHandlingMiddleware;