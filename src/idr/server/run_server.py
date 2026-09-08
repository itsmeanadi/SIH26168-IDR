"""Server runner script supporting HTTP and HTTPS with auto-generated local certs."""

import argparse
import os
import sys
import uvicorn
from idr.server.cert import generate_self_signed_cert


def run():
    parser = argparse.ArgumentParser(description="IDR Real-Time Navigation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS using self-signed certificate")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    ssl_keyfile = None
    ssl_certfile = None

    if args.ssl:
        cert_p, key_p = generate_self_signed_cert()
        ssl_certfile = str(cert_p)
        ssl_keyfile = str(key_p)
        print("=" * 70)
        print("  [SECURE HTTPS MODE ENABLED]")
        print(f"  Cert: {ssl_certfile}")
        print(f"  Key:  {ssl_keyfile}")
        print(f"  Mobile URL: https://<YOUR_LAPTOP_IP>:{args.port}")
        print("  (Accept the self-signed certificate warning on your mobile browser)")
        print("=" * 70)
    else:
        print("=" * 70)
        print("  [HTTP MODE (Standard LAN)]")
        print(f"  Mobile URL: http://<YOUR_LAPTOP_IP>:{args.port}")
        print("  Note: For Chrome on Android hardware IMU/GPS, enable:")
        print("  chrome://flags/#unsafely-treat-insecure-origin-as-secure")
        print("  Or run with --ssl for native HTTPS secure context.")
        print("=" * 70)

    uvicorn.run(
        "idr.server.app:app",
        host=args.host,
        port=args.port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
