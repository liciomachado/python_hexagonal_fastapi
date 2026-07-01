import unittest
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.error_handlers import INTERNAL_SERVER_ERROR_MESSAGE, register_error_handlers
from app.api.middleware.correlation_id import CORRELATION_ID_HEADER, CorrelationIdMiddleware
from app.api.result_utils import unwrap_result
from app.core.utils.result import BadRequestError, NotFoundError, Result


@asynccontextmanager
async def _test_lifespan(_app: FastAPI):
    yield


def _create_test_app() -> FastAPI:
    app = FastAPI(lifespan=_test_lifespan)
    app.add_middleware(CorrelationIdMiddleware)
    register_error_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/bad-request")
    async def bad_request():
        raise BadRequestError("invalid input")

    @app.get("/not-found")
    async def not_found():
        raise NotFoundError("not found")

    @app.get("/unexpected")
    async def unexpected():
        raise RuntimeError("boom")

    @app.get("/http-500")
    async def http_500():
        raise HTTPException(status_code=500, detail="internal failure")

    @app.get("/result-string-error")
    async def result_string_error():
        return unwrap_result(Result.Err("unexpected service failure"))

    return app


class ErrorHandlersAndCorrelationIdTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_create_test_app(), raise_server_exceptions=False)

    def test_success_returns_correlation_header(self):
        response = self.client.get("/ok")

        self.assertEqual(response.status_code, 200)
        self.assertIn(CORRELATION_ID_HEADER, response.headers)
        self.assertTrue(response.headers[CORRELATION_ID_HEADER])

    def test_propagates_incoming_correlation_id(self):
        correlation_id = "test-correlation-id-123"
        response = self.client.get("/ok", headers={CORRELATION_ID_HEADER: correlation_id})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get(CORRELATION_ID_HEADER), correlation_id)

    def test_bad_request_returns_400(self):
        response = self.client.get("/bad-request")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "invalid input")
        self.assertIn(CORRELATION_ID_HEADER, response.headers)

    def test_not_found_returns_404(self):
        response = self.client.get("/not-found")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "not found")
        self.assertIn(CORRELATION_ID_HEADER, response.headers)

    def test_unexpected_exception_returns_500_with_trace_id(self):
        response = self.client.get("/unexpected")

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["message"], INTERNAL_SERVER_ERROR_MESSAGE)
        self.assertEqual(body["traceId"], response.headers.get(CORRELATION_ID_HEADER))

    def test_http_500_returns_standard_payload(self):
        response = self.client.get("/http-500")

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["message"], INTERNAL_SERVER_ERROR_MESSAGE)
        self.assertEqual(body["traceId"], response.headers.get(CORRELATION_ID_HEADER))

    def test_result_string_error_returns_500_with_trace_id(self):
        response = self.client.get("/result-string-error")

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["message"], INTERNAL_SERVER_ERROR_MESSAGE)
        self.assertEqual(body["traceId"], response.headers.get(CORRELATION_ID_HEADER))


if __name__ == "__main__":
    unittest.main()
