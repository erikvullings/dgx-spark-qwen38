"""The box's key on every route of a vLLM lane's engine, as the SGLang lanes have it.

vLLM checks --api-key only on the paths under /v1, /v2, /inference and /cohere (its
AuthenticationMiddleware's GUARDED_PREFIX). /invocations, the SageMaker router that every
vLLM 0.29 server registers and that runs chat and completions, answered without the key,
and so did /tokenize, /detokenize and the elastic-EP routes (read in vLLM 0.29.0's source,
2026-10-14). On a box whose engine listens beyond loopback (ENGINE_BIND), anyone on the
network could generate on it. The SGLang lanes refuse everything but /health and /metrics
without the key: loaded with `--middleware vllm_engine_auth.RequireKey`, this makes the
vLLM lane refuse the same.

The key is read once, at start, from the file the unit mounts (QWEN38_KEY_FILE; the
engine is restarted to take a new key, as --config's is). It is compared the way vLLM
compares it, sha256 digests in constant time. Pure ASGI, like vLLM's own middleware: a
streamed answer and a client's disconnect pass through untouched.
"""
import hashlib
import json
import os
import secrets

# What the cockpit and the installers read without the key, on every lane.
OPEN = ("/health", "/metrics")


def _key_digest(path):
    with open(path, encoding="utf-8") as f:
        key = f.readline().strip()
    if not key:
        raise RuntimeError(f"{path} holds no key: an engine nobody can authenticate to stays down")
    return hashlib.sha256(key.encode("utf-8")).digest()


class RequireKey:
    def __init__(self, app):
        self.app = app
        self.digest = _key_digest(os.environ.get("QWEN38_KEY_FILE", "/out/api-key"))

    def authorized(self, scope):
        for name, value in scope.get("headers") or ():
            if name.lower() == b"authorization":
                scheme, _, param = value.decode("latin-1").partition(" ")
                return scheme.lower() == "bearer" and secrets.compare_digest(
                    hashlib.sha256(param.encode("utf-8")).digest(), self.digest)
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        root, path = scope.get("root_path", ""), scope["path"]
        if root and path.startswith(root):
            path = path[len(root):]
        if path in OPEN or path.startswith("/metrics/") or self.authorized(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": "Unauthorized"}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
