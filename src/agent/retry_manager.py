import logging

logger = logging.getLogger(__name__)

class RetryManager:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.attempts = []
        
    def add_attempt(self, config, error=None, fix_applied=None):
        attempt = {
            "attempt_no": len(self.attempts) + 1,
            "config": config.copy(),
            "error": error,
            "fix_applied": fix_applied
        }
        self.attempts.append(attempt)
        return attempt
        
    def should_retry(self):
        return len(self.attempts) < self.max_retries
        
    def get_summary(self):
        if not self.attempts:
            return "No attempts made."
            
        summary = f"Total attempts: {len(self.attempts)}\n"
        for i, att in enumerate(self.attempts):
            res = "Success" if not att["error"] else f"Error: {att['error']}"
            fix = f", Fix: {att['fix_applied']}" if att['fix_applied'] else ""
            summary += f"- Attempt {i+1}: {res}{fix}\n"
        return summary
