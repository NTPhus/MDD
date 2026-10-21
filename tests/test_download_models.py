"""Pinned downloads repair corrupt checkouts without trusting local candidates."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_models


def metadata(data, *, lfs=False):
    expected = {"size": len(data)}
    if lfs:
        expected["lfs"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    else:
        expected["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return expected


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.json"
        self.url = "https://huggingface.co/example/model/resolve/0123456789abcdef/config.json"
        self.output = patch("builtins.print")
        self.output.start()
        self.addCleanup(self.output.stop)

    def test_valid_existing_file_avoids_network(self):
        data = b'{"model_type": "wav2vec2"}\n'
        self.path.write_bytes(data)
        with patch.object(download_models, "urlopen") as open_url:
            result = download_models.download_file(self.url, self.path, metadata(data))
        open_url.assert_not_called()
        self.assertEqual(result["sha256"], hashlib.sha256(data).hexdigest())

    def test_crlf_checkout_is_replaced_only_after_byte_exact_download(self):
        data = b'{\n  "model_type": "wav2vec2"\n}\n'
        checkout = data.replace(b"\n", b"\r\n")
        self.path.write_bytes(checkout)

        def open_url(request, **kwargs):
            self.assertEqual(self.path.read_bytes(), checkout)
            return Response(data)

        with patch.object(download_models, "urlopen", side_effect=open_url):
            result = download_models.download_file(self.url, self.path, metadata(data))
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(result["filename"], "config.json")
        self.assertFalse(self.path.with_name("config.json.part").exists())

    def test_failed_replacement_preserves_original_and_rejects_same_size_corruption(self):
        original, data = b"original\r\n", b"correct\n"
        self.path.write_bytes(original)
        with patch.object(download_models, "urlopen", return_value=Response(b"corrupt\n")):
            with self.assertRaisesRegex(ValueError, "Git blob SHA-1"):
                download_models.download_file(self.url, self.path, metadata(data))
        self.assertEqual(self.path.read_bytes(), original)

    def test_pinned_lfs_weights_resume_and_verify_sha256(self):
        data = b"pinned model weights"
        partial = self.path.with_name("config.json.part")
        partial.write_bytes(data[:7])
        response = Response(data[7:], status=206, content_range=f"bytes 7-{len(data) - 1}/{len(data)}")
        with patch.object(download_models, "urlopen", return_value=response) as open_url:
            result = download_models.download_file(self.url, self.path, metadata(data, lfs=True))
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, self.url)
        self.assertEqual(request.get_header("Range"), "bytes=7-")
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(result["hub_sha256"], hashlib.sha256(data).hexdigest())

    def test_corrupt_complete_or_oversized_partial_restarts_without_range(self):
        data = b"correct\n"
        partial = self.path.with_name("config.json.part")
        for candidate in (b"corrupt\n", b"oversized corrupt data"):
            with self.subTest(candidate=candidate):
                partial.write_bytes(candidate)
                with patch.object(download_models, "urlopen", return_value=Response(data)) as open_url:
                    download_models.download_file(self.url, self.path, metadata(data))
                self.assertIsNone(open_url.call_args.args[0].get_header("Range"))
                self.assertEqual(self.path.read_bytes(), data)
                self.path.unlink()

    def test_corrupt_resumed_weights_retry_once_from_zero(self):
        data = b"correct model weights"
        self.path.with_name("config.json.part").write_bytes(b"broken ")
        responses = [Response(data[7:], status=206, content_range=f"bytes 7-{len(data) - 1}/{len(data)}"),
                     Response(data)]
        with patch.object(download_models, "urlopen", side_effect=responses) as open_url:
            download_models.download_file(self.url, self.path, metadata(data, lfs=True))
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(open_url.call_args_list[0].args[0].get_header("Range"), "bytes=7-")
        self.assertIsNone(open_url.call_args_list[1].args[0].get_header("Range"))
        self.assertEqual(self.path.read_bytes(), data)

    def test_invalid_resume_range_preserves_original(self):
        original, data = b"original\r\n", b"correct model weights"
        self.path.write_bytes(original)
        partial = self.path.with_name("config.json.part")
        partial.write_bytes(data[:7])
        response = Response(data[7:], status=206, content_range=f"bytes 0-{len(data) - 1}/{len(data)}")
        with patch.object(download_models, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "resume offset"):
                download_models.download_file(self.url, self.path, metadata(data, lfs=True))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(partial.read_bytes(), data[:7])


if __name__ == "__main__":
    unittest.main()
