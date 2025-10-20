from loguru import logger

logger.add("logs.log", rotation="1 MB", level="INFO")
