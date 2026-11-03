"""
Serve this folder locally and open index.html in the browser.
index.html fetches pong.py and opponentNN.py over HTTP, so it can't be opened as a file.
Usage: python serve.py [--port PORT]
"""
import argparse
import http.server
import os
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Serve index.html locally.")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def end_headers(self):
        # Always serve fresh files so edits to the .py files show up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as httpd:
    url = f"http://127.0.0.1:{args.port}/index.html"
    print(f"Serving {url}")
    print("Ctrl+C to stop")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
