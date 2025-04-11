import logging
import traceback

LOG_TRACE_LEVEL = logging.DEBUG // 2
logging.addLevelName(LOG_TRACE_LEVEL, "TRACE")

class CustomLogger(logging.Logger):
    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_TRACE_LEVEL):
            self._log(LOG_TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = CustomLogger.trace
logging.setLoggerClass(CustomLogger)


# Function to get a pre-configured logger
def get_logger(name: str = __name__) -> CustomLogger:
        return logging.getLogger(name)  # Returns an instance of CustomLogger

class StackTraceFormatter(logging.Formatter):
    def format(self, record):
        # Format the original log message
        message = super().format(record)
        # Append the stack trace if available
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        else:
            # Add the current stack trace for non-exception logs
            stack = traceback.format_stack()
            message += "\n" + "".join(stack[:-1])  # Exclude the current log call
        return message

# # Configure the logger
# log_format = "[%(asctime)s] [%(levelname)s] - %(name)s (%(pathname)s:%(lineno)d): %(message)s"
# handler = logging.StreamHandler()
# handler.setFormatter(StackTraceFormatter(log_format))
# logging.getLogger().addHandler(handler)
# logging.getLogger().setLevel(logging.DEBUG)