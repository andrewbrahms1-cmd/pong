"""
Written by claude code, for local testing purposes

Serve this folder locally and open one of the pages in the browser.
Usage: python serve.py [page] [--port PORT]
  e.g. python serve.py              (opens evolve.html)
       python serve.py nnplay
       python serve.py pong.html --port 8080
"""
import argparse
import http.server
import os
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Serve the pong pages locally.")
parser.add_argument("page", nargs="?", default="evolve.html", help="page to open (default: evolve.html)")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

page = args.page if args.page.endswith(".html") else args.page + ".html"
pages = sorted(f for f in os.listdir(HERE) if f.endswith(".html"))
if page not in pages:
    parser.error(f"{page} not found; available pages: {', '.join(pages)}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def end_headers(self):
        # Always serve fresh files so edits to the .py files show up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as httpd:
    print("Pages:")
    for p in pages:
        print(f"  http://127.0.0.1:{args.port}/{p}")
    print("Ctrl+C to stop")
    webbrowser.open(f"http://127.0.0.1:{args.port}/{page}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
