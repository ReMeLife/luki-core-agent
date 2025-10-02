#!/usr/bin/env python3
"""
Full LUKi Core Agent startup script with complete functionality.
Includes all ML modules, context engineering, and personality building.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    # Critical dependencies for LUKi functionality
    critical_deps = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'langchain': 'LangChain',
        'httpx': 'HTTPX',
        'sentence_transformers': 'Sentence Transformers',
        'chromadb': 'ChromaDB',
        'openai': 'OpenAI'
    }
    
    for dep_name, display_name in critical_deps.items():
        try:
            module = __import__(dep_name)
            if hasattr(module, '__version__'):
                logger.info(f"{display_name} version: {module.__version__}")
            else:
                logger.info(f"{display_name} available")
        except ImportError:
            logger.error(f"CRITICAL: {display_name} not found")
            missing_deps.append(dep_name)
        except Exception as e:
            logger.warning(f"{display_name} import issue: {e}")
            missing_deps.append(dep_name)
    
    if missing_deps:
        logger.error(f"CRITICAL DEPENDENCIES MISSING: {missing_deps}")
        logger.error("LUKi Core Agent CANNOT function without these dependencies!")
        return False
    
    logger.info("✅ ALL CRITICAL DEPENDENCIES AVAILABLE - Full LUKi functionality enabled")
    return True

def start_full_server():
    """Start the full LUKi Core Agent server"""
    try:
        logger.info("="*50)
        logger.info("LUKi Core Agent - Full Deployment v2")
        logger.info("="*50)
        
        port = int(os.getenv("PORT", "8080"))
        logger.info(f"Starting full server on port {port}")
        
        # Check dependencies first - FAIL HARD if missing
        if not check_dependencies():
            logger.error("DEPLOYMENT FAILED: Critical dependencies missing!")
            logger.error("LUKi Core Agent requires ALL dependencies for proper functionality")
            logger.error("Please check requirements-railway.txt and rebuild the deployment")
            sys.exit(1)
        
        # Try to import and start the full LUKi agent
        try:
            from luki_agent.main import app
            logger.info("Successfully imported full LUKi agent")
            
            import uvicorn
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                workers=1
            )
            
        except ImportError as e:
            logger.error(f"Failed to import full LUKi agent: {e}")
            logger.info("Attempting to start with dev_api...")
            
            try:
                from luki_agent.dev_api import app
                logger.info("Successfully imported dev_api")
                
                import uvicorn
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=port,
                    log_level="info",
                    workers=1
                )
                
            except ImportError as e2:
                logger.error(f"Failed to import dev_api: {e2}")
                return start_fallback_server(port)
                
    except Exception as e:
        logger.error(f"Failed to start full server: {e}")
        logger.error(traceback.format_exc())
        return start_fallback_server(port)

def start_fallback_server(port: int):
    """Start minimal fallback server if full server fails"""
    logger.info("Starting fallback minimal server...")
    
    try:
        # Import standalone API as fallback
        sys.path.insert(0, str(Path(__file__).parent))
        from standalone_api import app
        
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start fallback server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        start_full_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
