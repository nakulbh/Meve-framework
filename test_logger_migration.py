#!/usr/bin/env python3
"""
Quick test to verify logger migration is working
"""

# Test direct logger import
from meve.utils import get_logger

logger = get_logger("test_migration")

logger.info("Testing logger migration")
logger.debug("This is a debug message")
logger.warning("This is a warning")
logger.success("Logger migration successful!")

print("\n✅ All logger tests passed!")
