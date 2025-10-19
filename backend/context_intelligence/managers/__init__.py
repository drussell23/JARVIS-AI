"""
Context Intelligence Managers
==============================

Managers for system state and resource management
"""

from .space_state_manager import (
    SpaceStateManager,
    SpaceValidator,
    SpaceTransitionHandler,
    SpaceState,
    WindowState,
    WindowInfo,
    SpaceStateInfo,
    EdgeCaseResult,
    get_space_state_manager,
    initialize_space_state_manager
)

from .window_capture_manager import (
    WindowCaptureManager,
    PermissionChecker,
    ImageProcessor,
    CaptureRetryHandler,
    CaptureStatus,
    CaptureResult,
    WindowBounds,
    get_window_capture_manager,
    initialize_window_capture_manager
)

from .system_state_manager import (
    SystemStateManager,
    YabaiHealthChecker,
    DisplayStateDetector,
    SystemRecoveryHandler,
    YabaiState,
    DisplayState,
    SystemHealth,
    YabaiStatus,
    DisplayStatus,
    SystemStateInfo,
    get_system_state_manager,
    initialize_system_state_manager
)

from .api_network_manager import (
    APINetworkManager,
    APIHealthChecker,
    NetworkDetector,
    ImageOptimizer,
    RetryHandler,
    APIState,
    NetworkState,
    ImageOptimizationStatus,
    APIStatus,
    NetworkStatus,
    ImageOptimizationResult,
    RetryResult,
    get_api_network_manager,
    initialize_api_network_manager
)

from .error_handling_matrix import (
    ErrorHandlingMatrix,
    FallbackChain,
    PartialResultAggregator,
    ErrorRecoveryStrategy,
    ErrorMessageGenerator,
    ExecutionPriority,
    ExecutionStatus,
    ResultQuality,
    MethodResult,
    ExecutionReport,
    MethodDefinition,
    get_error_handling_matrix,
    initialize_error_handling_matrix
)

from .capture_strategy_manager import (
    CaptureStrategyManager,
    CaptureCache,
    CachedCapture,
    get_capture_strategy_manager,
    initialize_capture_strategy_manager
)

from .ocr_strategy_manager import (
    OCRStrategyManager,
    OCRCache,
    CachedOCR,
    OCRResult,
    ClaudeVisionOCR,
    TesseractOCR,
    ImageHasher,
    ImageMetadataExtractor,
    get_ocr_strategy_manager,
    initialize_ocr_strategy_manager
)

from .response_strategy_manager import (
    ResponseStrategyManager,
    ResponseQuality,
    ExtractedDetail,
    ResponseAnalysis,
    EnhancedResponse,
    DetailExtractor,
    SpecificityScorer,
    ActionableFormatter,
    ResponseEnhancer,
    get_response_strategy_manager,
    initialize_response_strategy_manager
)

from .context_aware_response_manager import (
    ContextAwareResponseManager,
    ConversationTracker,
    ContextInjector,
    ImplicitContextResolver,
    ContextType,
    ConversationTurn,
    EntityReference,
    ConversationContext,
    ContextEnrichment,
    get_context_aware_response_manager,
    initialize_context_aware_response_manager
)

from .proactive_suggestion_manager import (
    ProactiveSuggestionManager,
    PatternAnalyzer,
    SuggestionGenerator,
    SuggestionRanker,
    SuggestionFormatter,
    SuggestionType,
    Suggestion,
    SuggestionResult,
    get_proactive_suggestion_manager,
    initialize_proactive_suggestion_manager
)

from .confidence_manager import (
    ConfidenceManager,
    ConfidenceCalculator,
    UncertaintyFormatter,
    ConfidenceVisualIndicator,
    ConfidenceLevel,
    ConfidenceScore,
    ConfidenceFormattedResponse,
    get_confidence_manager,
    initialize_confidence_manager
)

from .multi_monitor_manager import (
    MultiMonitorManager,
    MonitorDetector,
    MonitorSpaceMapper,
    MonitorReferenceResolver,
    MonitorPosition,
    MonitorInfo,
    MonitorLayout,
    SpaceMonitorMapping,
    get_multi_monitor_manager,
    initialize_multi_monitor_manager
)

__all__ = [
    # Space State Management
    'SpaceStateManager',
    'SpaceValidator',
    'SpaceTransitionHandler',
    'SpaceState',
    'WindowState',
    'WindowInfo',
    'SpaceStateInfo',
    'EdgeCaseResult',
    'get_space_state_manager',
    'initialize_space_state_manager',

    # Window Capture Management
    'WindowCaptureManager',
    'PermissionChecker',
    'ImageProcessor',
    'CaptureRetryHandler',
    'CaptureStatus',
    'CaptureResult',
    'WindowBounds',
    'get_window_capture_manager',
    'initialize_window_capture_manager',

    # System State Management
    'SystemStateManager',
    'YabaiHealthChecker',
    'DisplayStateDetector',
    'SystemRecoveryHandler',
    'YabaiState',
    'DisplayState',
    'SystemHealth',
    'YabaiStatus',
    'DisplayStatus',
    'SystemStateInfo',
    'get_system_state_manager',
    'initialize_system_state_manager',

    # API & Network Management
    'APINetworkManager',
    'APIHealthChecker',
    'NetworkDetector',
    'ImageOptimizer',
    'RetryHandler',
    'APIState',
    'NetworkState',
    'ImageOptimizationStatus',
    'APIStatus',
    'NetworkStatus',
    'ImageOptimizationResult',
    'RetryResult',
    'get_api_network_manager',
    'initialize_api_network_manager',

    # Error Handling Matrix
    'ErrorHandlingMatrix',
    'FallbackChain',
    'PartialResultAggregator',
    'ErrorRecoveryStrategy',
    'ErrorMessageGenerator',
    'ExecutionPriority',
    'ExecutionStatus',
    'ResultQuality',
    'MethodResult',
    'ExecutionReport',
    'MethodDefinition',
    'get_error_handling_matrix',
    'initialize_error_handling_matrix',

    # Capture Strategy Management
    'CaptureStrategyManager',
    'CaptureCache',
    'CachedCapture',
    'get_capture_strategy_manager',
    'initialize_capture_strategy_manager',

    # OCR Strategy Management
    'OCRStrategyManager',
    'OCRCache',
    'CachedOCR',
    'OCRResult',
    'ClaudeVisionOCR',
    'TesseractOCR',
    'ImageHasher',
    'ImageMetadataExtractor',
    'get_ocr_strategy_manager',
    'initialize_ocr_strategy_manager',

    # Response Strategy Management
    'ResponseStrategyManager',
    'ResponseQuality',
    'ExtractedDetail',
    'ResponseAnalysis',
    'EnhancedResponse',
    'DetailExtractor',
    'SpecificityScorer',
    'ActionableFormatter',
    'ResponseEnhancer',
    'get_response_strategy_manager',
    'initialize_response_strategy_manager',

    # Context-Aware Response Management
    'ContextAwareResponseManager',
    'ConversationTracker',
    'ContextInjector',
    'ImplicitContextResolver',
    'ContextType',
    'ConversationTurn',
    'EntityReference',
    'ConversationContext',
    'ContextEnrichment',
    'get_context_aware_response_manager',
    'initialize_context_aware_response_manager',

    # Proactive Suggestion Management
    'ProactiveSuggestionManager',
    'PatternAnalyzer',
    'SuggestionGenerator',
    'SuggestionRanker',
    'SuggestionFormatter',
    'SuggestionType',
    'Suggestion',
    'SuggestionResult',
    'get_proactive_suggestion_manager',
    'initialize_proactive_suggestion_manager',

    # Confidence Management
    'ConfidenceManager',
    'ConfidenceCalculator',
    'UncertaintyFormatter',
    'ConfidenceVisualIndicator',
    'ConfidenceLevel',
    'ConfidenceScore',
    'ConfidenceFormattedResponse',
    'get_confidence_manager',
    'initialize_confidence_manager',

    # Multi-Monitor Management
    'MultiMonitorManager',
    'MonitorDetector',
    'MonitorSpaceMapper',
    'MonitorReferenceResolver',
    'MonitorPosition',
    'MonitorInfo',
    'MonitorLayout',
    'SpaceMonitorMapping',
    'get_multi_monitor_manager',
    'initialize_multi_monitor_manager',
]
