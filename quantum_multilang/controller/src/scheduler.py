#!/usr/bin/env python3
"""
Task Scheduler for Quantum Field Multi-Language Architecture

This module manages the scheduling of tasks across different language components,
optimizing for performance, coherence, and phi-harmonic load balancing.
"""

import time
import threading
import heapq
import logging
from enum import Enum
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_scheduler")

# Import sacred constants if available
try:
    import sacred_constants as sc
    PHI = sc.PHI
    LAMBDA = sc.LAMBDA
    PHI_PHI = sc.PHI_PHI
except ImportError:
    logger.warning("sacred_constants module not found. Using default values.")
    PHI = 1.618033988749895
    LAMBDA = 0.618033988749895
    PHI_PHI = 2.1784575679375995

class TaskPriority(Enum):
    """Task priority levels using phi-harmonic values."""
    CRITICAL = PHI_PHI  # ~2.18
    HIGH = PHI  # 1.618
    NORMAL = 1.0
    LOW = LAMBDA  # ~0.618
    BACKGROUND = LAMBDA * LAMBDA  # ~0.382

@dataclass(order=True)
class Task:
    """
    A task to be executed by the scheduler, with phi-harmonic priority.
    """
    priority: float
    timestamp: float
    id: int = field(compare=False)
    func: Callable = field(compare=False)
    args: Tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    language: str = field(default="python", compare=False)
    timeout: Optional[float] = field(default=None, compare=False)
    result: Any = field(default=None, compare=False)
    error: Exception = field(default=None, compare=False)
    completed: bool = field(default=False, compare=False)
    
    def execute(self):
        """Execute the task and store the result or error."""
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.completed = True
            return self.result
        except Exception as e:
            self.error = e
            self.completed = True
            logger.error(f"Task {self.id} failed: {e}")
            return None

class TaskScheduler:
    """
    Scheduler that manages task execution across different language components
    using phi-harmonic principles for load balancing and prioritization.
    """
    
    def __init__(self, worker_count=4):
        """
        Initialize the task scheduler.
        
        Args:
            worker_count: Number of worker threads
        """
        self.worker_count = worker_count
        self.task_queue = []  # Priority queue
        self.completed_tasks = {}
        self.next_task_id = 0
        self.queue_lock = threading.RLock()
        self.running = False
        self.workers = []
        self.language_load = {}  # Track load per language
        
        # Phi-harmonic load balancing parameters
        self.phi = PHI
        self.lambda_val = LAMBDA
        
        # Performance metrics for auto-tuning
        self.language_performance = {
            "python": 1.0,  # Baseline
            "rust": 1.5,    # Estimated relative performance
            "cpp": 1.4,     # Estimated relative performance
            "julia": 1.2,   # Estimated relative performance
            "go": 1.1,      # Estimated relative performance
            "wasm": 0.8,    # Estimated relative performance
            "zig": 1.3      # Estimated relative performance
        }
    
    def start(self):
        """Start the scheduler workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        self.workers = []
        for i in range(self.worker_count):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.workers.append(thread)
        
        logger.info(f"Started {self.worker_count} worker threads")
    
    def stop(self):
        """Stop the scheduler workers."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        logger.info("Stopped worker threads")
    
    def schedule_task(self, func, args=None, kwargs=None, priority=TaskPriority.NORMAL, 
                      language="python", timeout=None):
        """
        Schedule a task for execution.
        
        Args:
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority
            language: Preferred language for execution
            timeout: Maximum execution time in seconds
            
        Returns:
            Task ID for retrieving results later
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        
        with self.queue_lock:
            # Generate a task ID
            task_id = self.next_task_id
            self.next_task_id += 1
            
            # Convert priority enum to value if needed
            priority_value = priority.value if isinstance(priority, TaskPriority) else float(priority)
            
            # Create a task
            task = Task(
                priority=priority_value,
                timestamp=time.time(),
                id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                language=language,
                timeout=timeout
            )
            
            # Add to the priority queue
            heapq.heappush(self.task_queue, task)
            
            logger.debug(f"Scheduled task {task_id} with priority {priority_value}")
            
            return task_id
    
    def get_result(self, task_id, wait=False, timeout=None):
        """
        Get the result of a task.
        
        Args:
            task_id: The ID of the task
            wait: Whether to wait for the task to complete
            timeout: Maximum time to wait in seconds
            
        Returns:
            The task result, or None if not available
        """
        start_time = time.time()
        
        while True:
            # Check if the task is completed
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.error:
                    raise task.error
                return task.result
            
            # If not waiting, return None
            if not wait:
                return None
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Wait a bit before checking again
            time.sleep(0.01)
    
    def _worker_loop(self, worker_id):
        """Worker thread main loop."""
        logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            task = None
            
            # Get a task from the queue
            with self.queue_lock:
                if self.task_queue:
                    task = heapq.heappop(self.task_queue)
                    
                    # Update language load
                    lang = task.language
                    self.language_load[lang] = self.language_load.get(lang, 0) + 1
            
            if task:
                # Execute the task
                logger.debug(f"Worker {worker_id} executing task {task.id}")
                
                start_time = time.time()
                result = task.execute()
                execution_time = time.time() - start_time
                
                # Update language performance metrics
                self._update_performance_metrics(task.language, execution_time)
                
                # Store the result
                with self.queue_lock:
                    self.completed_tasks[task.id] = task
                    
                    # Update language load
                    self.language_load[task.language] = max(0, self.language_load.get(task.language, 0) - 1)
                
                logger.debug(f"Worker {worker_id} completed task {task.id}")
            else:
                # No tasks available, sleep a bit
                time.sleep(0.01)
    
    def _update_performance_metrics(self, language, execution_time):
        """Update performance metrics for a language."""
        # Simple exponential moving average
        alpha = 0.1  # Smoothing factor
        old_perf = self.language_performance.get(language, 1.0)
        
        # Inverse of execution time (higher is better)
        new_perf_raw = 1.0 / max(0.001, execution_time)
        
        # Normalize to Python baseline
        python_perf = self.language_performance.get("python", 1.0)
        new_perf = new_perf_raw / python_perf if python_perf > 0 else new_perf_raw
        
        # Update using EMA
        updated_perf = old_perf * (1 - alpha) + new_perf * alpha
        
        self.language_performance[language] = updated_perf
    
    def get_optimal_language(self, task_size):
        """
        Determine the optimal language for a task based on size and current load.
        
        Args:
            task_size: Size metric for the task (e.g., field dimensions)
            
        Returns:
            The optimal language for the task
        """
        # Calculate a score for each language based on:
        # 1. Performance metrics
        # 2. Current load
        # 3. Task size suitability
        
        scores = {}
        
        for language, perf in self.language_performance.items():
            # Base score is performance
            score = perf
            
            # Adjust for current load (phi-weighted)
            load = self.language_load.get(language, 0)
            load_factor = self.phi ** (-load)  # Higher load reduces score
            score *= load_factor
            
            # Adjust for task size
            if task_size > 1000000:
                # Large tasks favor high-performance languages
                if language in ("rust", "cpp"):
                    score *= self.phi
            elif task_size < 10000:
                # Small tasks may be better in Python due to overhead
                if language == "python":
                    score *= self.lambda_val
            
            scores[language] = score
        
        # Find the language with the highest score
        if scores:
            optimal_language = max(scores.items(), key=lambda x: x[1])[0]
        else:
            optimal_language = "python"  # Default
        
        return optimal_language
    
    def get_scheduler_stats(self):
        """
        Get statistics about the scheduler state.
        
        Returns:
            A dictionary with scheduler statistics
        """
        with self.queue_lock:
            stats = {
                "pending_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "workers": self.worker_count,
                "running": self.running,
                "language_load": dict(self.language_load),
                "language_performance": dict(self.language_performance)
            }
            
            return stats

# Simple test
if __name__ == "__main__":
    def test_task(x, y, sleep_time=0.1):
        """Test task that simulates work."""
        time.sleep(sleep_time)
        return x + y
    
    # Create scheduler
    scheduler = TaskScheduler(worker_count=2)
    scheduler.start()
    
    try:
        # Schedule some tasks
        task1 = scheduler.schedule_task(test_task, args=(1, 2), priority=TaskPriority.HIGH)
        task2 = scheduler.schedule_task(test_task, args=(3, 4), priority=TaskPriority.NORMAL)
        task3 = scheduler.schedule_task(test_task, args=(5, 6), priority=TaskPriority.LOW)
        
        # Get results
        result1 = scheduler.get_result(task1, wait=True)
        result2 = scheduler.get_result(task2, wait=True)
        result3 = scheduler.get_result(task3, wait=True)
        
        print(f"Task 1 result: {result1}")
        print(f"Task 2 result: {result2}")
        print(f"Task 3 result: {result3}")
        
        # Get stats
        stats = scheduler.get_scheduler_stats()
        print(f"Scheduler stats: {stats}")
        
        # Test optimal language selection
        small_task_lang = scheduler.get_optimal_language(5000)
        large_task_lang = scheduler.get_optimal_language(2000000)
        
        print(f"Optimal language for small task: {small_task_lang}")
        print(f"Optimal language for large task: {large_task_lang}")
        
    finally:
        # Stop the scheduler
        scheduler.stop()