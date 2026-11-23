"""Tiny OpenAI-compatible stub server for the CPU-only smoke test.

It answers every chat completion with a fixed string (or the first of the
allowed answer options), so the whole SpaCE-GLUE pipeline can be exercised
without a GPU or a real model. The scores it produces are meaningless.

Only the Python standard library is used.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "smoke-test-stub"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}
            )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, status=404)
            return

        options = (request.get("structured_outputs") or {}).get("choice")
        answer = options[0] if options else "yes"
        self._send_json(
            {
                "id": "chatcmpl-smoke-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", MODEL_NAME),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    def log_message(self, format, *args):
        pass  # keep the job log readable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Stub server listening on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
