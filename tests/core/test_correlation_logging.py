import io
import logging
import unittest

from app.core.correlation_context import reset_correlation_id, set_correlation_id
from app.core.logging_config import CorrelationIdLogFilter


class CorrelationIdLogFilterTests(unittest.TestCase):
    def test_filter_injects_correlation_id_into_log_record(self):
        log_filter = CorrelationIdLogFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )

        self.assertEqual(log_filter.filter(record), True)
        self.assertEqual(record.correlation_id, "-")

        token = set_correlation_id("corr-abc-123")
        try:
            self.assertEqual(log_filter.filter(record), True)
            self.assertEqual(record.correlation_id, "corr-abc-123")
        finally:
            reset_correlation_id(token)

    def test_setup_logging_includes_correlation_id_in_formatted_output(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | [%(correlation_id)s] | %(message)s"
            )
        )
        handler.addFilter(CorrelationIdLogFilter())

        logger = logging.getLogger("tests.correlation_logging")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        token = set_correlation_id("trace-for-filter-test")
        try:
            logger.error("Unhandled exception [correlation_id=%s]", "trace-for-filter-test")
        finally:
            reset_correlation_id(token)

        output = stream.getvalue()
        self.assertIn("[trace-for-filter-test]", output)
        self.assertIn("Unhandled exception [correlation_id=trace-for-filter-test]", output)


if __name__ == "__main__":
    unittest.main()
