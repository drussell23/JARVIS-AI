"""Configuration package for the backend application.

This package provides configuration management for the backend application,
including settings for database connections, API configurations, environment
variables, and other application-wide parameters.

The configuration system supports multiple environments (development, testing,
production) and provides a centralized way to manage application settings.

Example:
    Import configuration components:
    
    >>> from config import settings
    >>> from config.database import DatabaseConfig
    >>> from config.api import APIConfig
"""

# Configuration package initialization
# This file makes the config directory a Python package