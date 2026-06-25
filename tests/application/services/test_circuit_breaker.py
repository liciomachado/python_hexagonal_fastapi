import unittest

from app.application.services.resilience.circuit_breaker import CircuitBreaker


class CircuitBreakerTests(unittest.TestCase):
    def test_starts_closed(self):
        breaker = CircuitBreaker(open_seconds=300)
        self.assertFalse(breaker.is_open())
        self.assertEqual(breaker.state(), "closed")
        self.assertIsNone(breaker.opened_until())

    def test_opens_and_closes_after_window(self):
        breaker = CircuitBreaker(open_seconds=300)
        breaker.open()
        self.assertTrue(breaker.is_open())
        self.assertIsNotNone(breaker.opened_until())

        breaker._opened_until = breaker._opened_until - 301
        self.assertFalse(breaker.is_open())


if __name__ == "__main__":
    unittest.main()
