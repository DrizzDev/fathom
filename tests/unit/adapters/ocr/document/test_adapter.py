from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fathom.adapters.ocr.document.adapter import DocumentAiOcr
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import DocumentAiConfiguration
from fathom.schemas.screens import ScreenCapture


class _StubClient:
    """
    Test double matching the Document AI client surface the adapter uses.

    Records the number of calls so the dispatch path can be asserted
    independently of the mapper's token extraction. Returns a frozen
    response wrapped in a :class:`SimpleNamespace` shaped like the
    real protobuf message — ``response.document`` is the only attribute
    the adapter reads.
    """

    def __init__(self, *, document: Any) -> None:
        """
        Initialise with the fake document payload returned on every call.
        The :class:`DocumentAiMapper` walks ``document.pages[*].tokens``
        so the fixture only needs to provide that shape.
        """

        self.__document = document
        self.calls: int = 0

    def process_document(self, *, request: Any) -> Any:
        """
        Increment the call counter and return the preconfigured response
        wrapped in a ``SimpleNamespace`` so attribute access mirrors the
        real protobuf message.
        """

        _ = request
        self.calls += 1
        return SimpleNamespace(document=self.__document)


class _FailingClient:
    """
    Test double that raises on ``process_document``. Drives the
    adapter's error-suppression path so a raw provider exception
    surfaces as an empty :class:`OcrResult` rather than propagating.
    """

    def process_document(self, *, request: Any) -> Any:
        """
        Raise a deterministic :class:`RuntimeError`. The adapter must
        catch this, log via ``logger.exception``, and return an empty
        :class:`OcrResult` so OCR — an optional perception enrichment
        — never breaks the surrounding run.
        """

        _ = request
        raise RuntimeError("documentai down")


class DocumentAiOcrAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :class:`DocumentAiOcr` request, response, and graceful-failure.

    The adapter runs the Document AI ``process_document`` call on a
    worker thread inside ``asyncio.wait_for``, then hands the response
    to :class:`DocumentAiMapper`. OCR is an optional perception
    enrichment: every failure mode (generic provider exception, budget
    timeout) is suppressed at the adapter boundary and returned as an
    empty :class:`OcrResult` so the caller can degrade gracefully.

    The tests cover: empty-document happy path, dispatch counting,
    provider-exception → empty result, budget timeout → empty result,
    and default-client construction (without a stub client, the adapter
    must instantiate the real Document AI client class).
    """

    @staticmethod
    def __configuration() -> DocumentAiConfiguration:
        """
        :class:`DocumentAiConfiguration` fixture with placeholder
        identifiers. The adapter feeds these into the request path
        attribute (``projects/.../locations/.../processors/...``).
        """

        return DocumentAiConfiguration(
            project="vision-478905",
            location="us",
            processor="proc-1",
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture with a PNG-magic prefix in the
        image payload. The stub client never inspects the bytes; the
        prefix is there to make the fixture look like a real PNG to
        any future check that gates on magic bytes.
        """

        return ScreenCapture(
            width=1000,
            height=2000,
            activity="app",
            image=b"\x89PNG\r\n\x1a\nfake",
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Permissive :class:`PerceptionBudget`. The timeout-mapping test
        overrides ``ocr`` to a tight 10ms so the worker thread misses
        the deadline.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    @staticmethod
    def __empty_document() -> SimpleNamespace:
        """
        Document AI document fixture with no pages. The mapper iterates
        ``document.pages`` so an empty list produces an empty token
        tuple, exercising the happy-path zero-token branch.
        """

        return SimpleNamespace(text="", pages=[])

    async def test_extract_returns_empty_result_for_empty_document(self) -> None:
        """
        An empty document yields an empty OcrResult with a recorded duration.
        """

        adapter = DocumentAiOcr(
            configuration=self.__configuration(),
            client=_StubClient(document=self.__empty_document()),
        )

        result = await adapter.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(result.tokens, ())
        self.assertGreaterEqual(result.duration, 0)
        self.assertIsNotNone(result.raw_response)
        self.assertIn('"text": ""', result.raw_response or "")

    async def test_extract_invokes_client_with_processor_path(self) -> None:
        """
        The adapter delegates the call to the Document AI client exactly once.
        """

        client = _StubClient(document=self.__empty_document())
        adapter = DocumentAiOcr(configuration=self.__configuration(), client=client)

        await adapter.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(client.calls, 1)

    async def test_extract_returns_empty_result_on_provider_exception(self) -> None:
        """
        A client exception must be suppressed at the adapter boundary
        and surface as an empty :class:`OcrResult`. Optional perception
        enrichments must never break the surrounding run.
        """

        adapter = DocumentAiOcr(
            configuration=self.__configuration(),
            client=_FailingClient(),
        )

        result = await adapter.extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(result.tokens, ())
        self.assertGreaterEqual(result.duration, 0)

    async def test_extract_returns_empty_result_on_budget_timeout(self) -> None:
        """
        A client call that exceeds the OCR budget must surface as an
        empty :class:`OcrResult` rather than propagating a TimeoutError.
        """

        class _BlockingClient:
            """
            Test double that blocks long enough to exceed the OCR budget.
            """

            def process_document(self, *, request: Any) -> Any:
                """
                Sleep past the budget on the worker thread.
                """

                _ = request
                import time as _time

                _time.sleep(0.5)
                return SimpleNamespace(document=SimpleNamespace(text="", pages=[]))

        adapter = DocumentAiOcr(
            configuration=self.__configuration(),
            client=_BlockingClient(),
        )
        budget = PerceptionBudget(ocr=10, local=10, localization=10)

        result = await adapter.extract(capture=self.__capture(), budget=budget)

        self.assertEqual(result.tokens, ())
        self.assertGreaterEqual(result.duration, 0)

    def test_build_default_client_when_omitted(self) -> None:
        """
        Omitting the client parameter must construct the default Document AI client.
        """

        with mock.patch(
            "fathom.adapters.ocr.document.adapter.documentai_v1.DocumentProcessorServiceClient",
            return_value=mock.MagicMock(),
        ) as fake_client_class:
            DocumentAiOcr(configuration=self.__configuration())

        self.assertTrue(fake_client_class.called)
        kwargs = fake_client_class.call_args.kwargs
        self.assertIn("credentials", kwargs)
        self.assertIsNone(kwargs["credentials"])

    def test_build_client_materializes_inline_dict_credentials(self) -> None:
        """Inline dict credentials are materialized into a :class:`service_account.Credentials` instance via ``from_service_account_info`` and passed to the gRPC client."""

        configuration = DocumentAiConfiguration(
            project="vision-478905",
            location="us",
            processor="proc-1",
            credentials={"type": "service_account", "project_id": "vision-478905"},
        )

        fake_credentials = mock.MagicMock(name="service_account_credentials")

        with (
            mock.patch(
                "fathom.adapters.ocr.document.adapter.service_account.Credentials.from_service_account_info",
                return_value=fake_credentials,
            ) as from_info,
            mock.patch(
                "fathom.adapters.ocr.document.adapter.documentai_v1.DocumentProcessorServiceClient",
                return_value=mock.MagicMock(),
            ) as fake_client_class,
        ):
            DocumentAiOcr(configuration=configuration)

        from_info.assert_called_once()
        self.assertIs(fake_client_class.call_args.kwargs["credentials"], fake_credentials)

    def test_build_client_materializes_file_path_credentials(self) -> None:
        """A file-path credentials value resolves through ``from_service_account_file`` and is forwarded to the gRPC client."""

        configuration = DocumentAiConfiguration(
            project="vision-478905",
            location="us",
            processor="proc-1",
            credentials="/tmp/key.json",  # nosec - test fixture
        )

        fake_credentials = mock.MagicMock(name="service_account_credentials")

        with (
            mock.patch(
                "fathom.adapters.ocr.document.adapter.Path.is_file",
                return_value=True,
            ),
            mock.patch(
                "fathom.adapters.ocr.document.adapter.service_account.Credentials.from_service_account_file",
                return_value=fake_credentials,
            ) as from_file,
            mock.patch(
                "fathom.adapters.ocr.document.adapter.documentai_v1.DocumentProcessorServiceClient",
                return_value=mock.MagicMock(),
            ) as fake_client_class,
        ):
            DocumentAiOcr(configuration=configuration)

        from_file.assert_called_once()
        self.assertIs(fake_client_class.call_args.kwargs["credentials"], fake_credentials)

    def test_build_client_returns_none_when_credentials_path_missing(self) -> None:
        """
        A file path that does not exist must collapse to ``credentials=None``
        instead of crashing — the adapter logs the miss and the gRPC client
        will surface a typed auth error if it later runs without credentials.
        """

        configuration = DocumentAiConfiguration(
            project="vision-478905",
            location="us",
            processor="proc-1",
            credentials="/does/not/exist.json",
        )

        with (
            mock.patch(
                "fathom.adapters.ocr.document.adapter.Path.is_file",
                return_value=False,
            ),
            mock.patch(
                "fathom.adapters.ocr.document.adapter.documentai_v1.DocumentProcessorServiceClient",
                return_value=mock.MagicMock(),
            ) as fake_client_class,
        ):
            DocumentAiOcr(configuration=configuration)

        self.assertIsNone(fake_client_class.call_args.kwargs["credentials"])
