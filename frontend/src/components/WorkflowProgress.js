import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const WorkflowProgress = ({ workflow, currentAction, onRetry, onCancel }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [retryCount, setRetryCount] = useState({});

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  if (!workflow) {
    return null;
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={16} />;
      case 'failed':
        return <XCircle className="text-red-500" size={16} />;
      case 'running':
        return <RefreshCw className="text-blue-500 animate-spin" size={16} />;
      case 'retry':
        return <RefreshCw className="text-orange-500" size={16} />;
      default:
        return <Clock className="text-gray-400" size={16} />;
    }
  };

  const getComplexityColor = (complexity) => {
    switch (complexity) {
      case 'simple':
        return 'text-green-500';
      case 'moderate':
        return 'text-yellow-500';
      case 'complex':
        return 'text-orange-500';
      case 'very_complex':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const completedCount = workflow.actions.filter(a => a.status === 'completed').length;
  const progress = (completedCount / workflow.total_actions) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="workflow-progress bg-gray-800 rounded-lg p-4 mt-4 border border-gray-700"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-white font-semibold">Workflow Progress</h3>
          <span className={`text-sm ${getComplexityColor(workflow.complexity)}`}>
            {workflow.complexity.replace('_', ' ')} workflow
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-400 hover:text-white transition-colors"
        >
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm text-gray-400">
            {completedCount} of {workflow.total_actions} actions
          </span>
          <span className="text-sm text-gray-400">{formatTime(elapsedTime)}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Estimated: {formatTime(workflow.estimated_duration)}
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-2 mt-3 border-t border-gray-700 pt-3">
              {workflow.actions.map((action, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`flex items-center justify-between p-2 rounded ${
                    index === currentAction ? 'bg-gray-700' : 'bg-gray-800/50'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {getStatusIcon(action.status)}
                    <span className={`text-sm ${
                      index === currentAction ? 'text-white' : 'text-gray-400'
                    }`}>
                      {action.description}
                    </span>
                    {action.status === 'retry' && (
                      <span className="text-xs text-orange-400">
                        (Retry {retryCount[index] || 1})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {action.duration && (
                      <span className="text-xs text-gray-500">
                        {action.duration}ms
                      </span>
                    )}
                    {action.status === 'failed' && onRetry && (
                      <button
                        onClick={() => {
                          onRetry(index);
                          setRetryCount(prev => ({ ...prev, [index]: (prev[index] || 0) + 1 }));
                        }}
                        className="text-xs text-blue-400 hover:text-blue-300"
                      >
                        Retry
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>

            {workflow.actions.some(a => a.status === 'failed') && (
              <div className="mt-3 p-2 bg-red-900/20 rounded border border-red-800">
                <p className="text-sm text-red-400">
                  {workflow.actions.find(a => a.status === 'failed')?.error || 'An error occurred'}
                </p>
                {onCancel && (
                  <button
                    onClick={onCancel}
                    className="text-xs text-red-400 hover:text-red-300 mt-1"
                  >
                    Cancel Workflow
                  </button>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default WorkflowProgress;