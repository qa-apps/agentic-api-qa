"""Test-suite isolation from live observability backends."""

import os


os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["QA_REPORT_SERVER_ENABLED"] = "false"
