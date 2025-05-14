# debug_tools.py
import functools
import time
import traceback
import logging
import os
import json
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vessco_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vessco")

# Type for function return value
T = TypeVar('T')

def debug_trace(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to trace function calls, parameters, and return values.
    Also captures and logs exceptions but allows them to propagate.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract basic information
        func_name = func.__name__
        module_name = func.__module__
        
        # Log function entry with parameters
        param_str = ""
        if args:
            param_str += f"Args: {args}"
        if kwargs:
            param_str += f", Kwargs: {kwargs}"
        
        logger.info(f"ENTER {module_name}.{func_name} - {param_str}")
        
        # Calculate execution time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log function exit with result
            execution_time = time.time() - start_time
            
            # Truncate result for logging if it's too large
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... [truncated]"
                
            logger.info(f"EXIT {module_name}.{func_name} - Time: {execution_time:.2f}s - Result: {result_str}")
            
            return result
            
        except Exception as e:
            # Log the exception but let it propagate
            execution_time = time.time() - start_time
            logger.error(f"ERROR in {module_name}.{func_name} - Time: {execution_time:.2f}s - Exception: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    return wrapper

def inspect_obj(obj: Any, max_depth: int = 3) -> None:
    """
    Recursively inspect an object's attributes and properties.
    Useful for debugging complex objects.
    """
    def _inspect(obj, depth=0, path=""):
        if depth >= max_depth:
            return
            
        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                current_path = f"{path}.{key}" if path else key
                value_type = type(value).__name__
                
                if isinstance(value, (str, int, float, bool, type(None))):
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type}): {value}")
                else:
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type})")
                    _inspect(value, depth + 1, current_path)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}['{key}']" if path else f"['{key}']"
                value_type = type(value).__name__
                
                if isinstance(value, (str, int, float, bool, type(None))):
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type}): {value}")
                else:
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type})")
                    _inspect(value, depth + 1, current_path)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                value_type = type(value).__name__
                
                if isinstance(value, (str, int, float, bool, type(None))):
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type}): {value}")
                else:
                    logger.info(f"{' ' * depth * 2}{current_path} ({value_type})")
                    _inspect(value, depth + 1, current_path)
    
    obj_type = type(obj).__name__
    logger.info(f"Inspecting object of type: {obj_type}")
    _inspect(obj)

def log_llm_interaction(prompt: str, response: str, duration: float, model: str = "unknown") -> None:
    """
    Log details of an interaction with the LLM.
    """
    # Create a log entry
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "duration": duration,
        "prompt": prompt,
        "response": response
    }
    
    # Log to file
    log_dir = "llm_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"llm_log_{time.strftime('%Y%m%d')}.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    # Also log a shorter version to the main log
    prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
    response_preview = response[:100] + "..." if len(response) > 100 else response
    
    logger.info(f"LLM Interaction - Model: {model}, Duration: {duration:.2f}s")
    logger.info(f"Prompt: {prompt_preview}")
    logger.info(f"Response: {response_preview}")

def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function's performance.
    Returns timing statistics and result.
    """
    # Run the function once to warm up
    func(*args, **kwargs)
    
    # Run multiple times to get performance metrics
    runs = 5
    times = []
    
    for _ in range(runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    logger.info(f"Benchmark for {func.__name__}:")
    logger.info(f"  Average time: {avg_time:.4f}s")
    logger.info(f"  Min time: {min_time:.4f}s")
    logger.info(f"  Max time: {max_time:.4f}s")
    
    return {
        "function": func.__name__,
        "average_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "result": result
    }

def monkey_patch_llm(llm_obj):
    """
    Monkey patch an LLM object to log all interactions.
    """
    original_invoke = llm_obj.invoke
    
    def patched_invoke(prompt, **kwargs):
        start_time = time.time()
        
        try:
            response = original_invoke(prompt, **kwargs)
            duration = time.time() - start_time
            
            # Log the interaction
            log_llm_interaction(
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                response=response.content if hasattr(response, 'content') else str(response),
                duration=duration,
                model=getattr(llm_obj, 'model_id', 'unknown')
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM invoke: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    # Replace the original method
    llm_obj.invoke = patched_invoke
    
    return llm_obj