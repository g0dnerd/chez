// Simple HTTP server for serving the Chez web interface
// Serves static files from zig-out/web/

const std = @import("std");

const port: u16 = 8080;
const web_dir = "zig-out/web";

pub fn main() !void {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const allocator = std.heap.page_allocator;

    const address = std.Io.net.IpAddress.parseIp4("127.0.0.1", port) catch unreachable;
    var server = try address.listen(io, .{
        .reuse_address = true,
    });
    defer server.deinit();

    std.log.info("Server listening on http://127.0.0.1:{d}", .{port});
    std.log.info("Serving files from {s}/", .{web_dir});

    while (true) {
        const conn = server.accept(io) catch |err| {
            std.log.err("Accept error: {}", .{err});
            continue;
        };
        defer conn.close(io);

        var reader_buf: [4096]u8 = undefined;
        var stream_reader = conn.reader(io, &reader_buf);

        var writer_buf: [4096]u8 = undefined;
        var stream_writer = conn.writer(io, &writer_buf);
        handleConnection(allocator, io, &stream_reader.interface, &stream_writer.interface) catch |err| {
            std.log.err("Connection error: {}", .{err});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, io: std.Io, r: *std.Io.Reader, w: *std.Io.Writer) !void {
    const request = try r.takeDelimiterExclusive('\n');

    if (request.len == 0) {
        return;
    }

    // Parse the request line
    var lines = std.mem.splitScalar(u8, request, '\n');
    const request_line = lines.first();

    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return;
    var path = parts.next() orelse return;

    // Only handle GET requests
    if (!std.mem.eql(u8, method, "GET")) {
        try sendResponse(w, "405 Method Not Allowed", "text/plain", "Method Not Allowed");
        return;
    }

    // Default to index.html
    if (std.mem.eql(u8, path, "/")) {
        path = "/index.html";
    }

    // Security: prevent directory traversal
    if (std.mem.indexOf(u8, path, "..") != null) {
        try sendResponse(w, "403 Forbidden", "text/plain", "Forbidden");
        return;
    }

    // Build file path
    const file_path = try std.fmt.allocPrint(allocator, "{s}{s}", .{ web_dir, path });
    defer allocator.free(file_path);

    // Try to open and read the file
    const file = std.Io.Dir.cwd().openFile(io, file_path, .{}) catch {
        try sendResponse(w, "404 Not Found", "text/plain", "Not Found");
        return;
    };
    defer file.close(io);

    const stat = try file.stat(io);
    const content = try allocator.alloc(u8, stat.size);
    defer allocator.free(content);
    var f_reader = file.reader(io, content);
    _ = try f_reader.interface.take(@intCast(stat.size));

    // Determine content type
    const content_type = getContentType(path);

    try sendResponseWithContent(w, "200 OK", content_type, content);
}

fn getContentType(path: []const u8) []const u8 {
    if (std.mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".css")) return "text/css; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".js")) return "application/javascript; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".wasm")) return "application/wasm";
    if (std.mem.endsWith(u8, path, ".json")) return "application/json";
    if (std.mem.endsWith(u8, path, ".png")) return "image/png";
    if (std.mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (std.mem.endsWith(u8, path, ".ico")) return "image/x-icon";
    return "application/octet-stream";
}

fn sendResponse(w: *std.Io.Writer, status: []const u8, content_type: []const u8, body: []const u8) !void {
    try sendResponseWithContent(w, status, content_type, body);
}

fn sendResponseWithContent(w: *std.Io.Writer, status: []const u8, content_type: []const u8, body: []const u8) !void {
    try w.print("HTTP/1.1 {s}\r\n", .{status});
    try w.print("Content-Type: {s}\r\n", .{content_type});
    try w.print("Content-Length: {d}\r\n", .{body.len});
    try w.writeAll("Access-Control-Allow-Origin: *\r\n");
    try w.writeAll("Cross-Origin-Opener-Policy: same-origin\r\n");
    try w.writeAll("Cross-Origin-Embedder-Policy: require-corp\r\n");
    try w.writeAll("Connection: close\r\n");
    try w.writeAll("\r\n");
    try w.writeAll(body);
    try w.flush();
}
