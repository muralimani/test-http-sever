import http.server
import socketserver

PORT = 8001

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    # Start the server and keep it running until you stop the script
    httpd.serve_forever()
