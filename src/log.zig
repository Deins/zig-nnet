const std = @import("std");
const ansi = @import("ansi_esc.zig");

const Self = @This();
pub const buff_size = 512;
pub const log_level: std.log.Level = switch (std.builtin.mode) {
    .Debug => .debug,
    .ReleaseSafe => .info,
    .ReleaseFast => .info,
    .ReleaseSmall => .info,
};
pub var log_ctx: Self = undefined; // TODO: init checks

out_mut: std.Thread.Mutex = .{},
out: std.io.BufferedWriter(buff_size, std.fs.File.Writer),
err: std.io.BufferedWriter(buff_size, std.fs.File.Writer),

pub fn init() void {
    log_ctx = .{
        .out = .{ .unbuffered_writer = std.io.getStdOut().writer() },
        .err = .{ .unbuffered_writer = std.io.getStdErr().writer() },
    };
    configure_console();
}

pub fn deinit() !void {
    try log_ctx.out.flush();
    try log_ctx.err.flush();
}

pub fn getCtx() *Self {
    return &log_ctx;
}

pub const scoped = std.log.scoped;

pub fn log(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@enumToInt(level) > @enumToInt(log_level)) return;
    const use_err = @enumToInt(level) <= @enumToInt(std.log.Level.warn);
    const flush = if (use_err) log_ctx.err.flush else log_ctx.out.flush;
    var out = if (use_err) log_ctx.err.writer() else log_ctx.out.writer();
    const outm = if (use_err) std.debug.getStderrMutex() else &log_ctx.out_mut;
    if (scope == .raw) {
        const lock = outm.acquire();
        out.print(format, args) catch @panic("Can't write log!");
        lock.release();
        return;
    }

    const color = switch (level) {
        .emerg => ansi.style.bold ++ ansi.style.underline ++ ansi.style.fg.col256(214) ++ ansi.style.bg.col256(52),
        .alert => ansi.style.bold ++ ansi.style.underline ++ ansi.style.fg.red,
        .crit => ansi.style.bold ++ ansi.style.fg.red,
        .err => ansi.style.fg.red,
        .warn => "" ++ ansi.style.fg.col256(214),
        .notice => ansi.style.bold,
        .info => "",
        .debug => comptime ansi.style.fg.col256(245) ++ ansi.style.italic,
    };
    //const scope_prefix = "(" ++ @tagName(scope) ++ "):\t";
    //const prefix = "[" ++ @tagName(level) ++ "] " ++ scope_prefix;
    const prefix = "";
    const lock = outm.acquire();
    const postfix = if (format.len > 1 and format[format.len-1] < ' ') "" else "\n";
    out.print(color ++ prefix ++ format ++ ansi.style.reset ++ postfix, args) catch @panic("Can't write log!");
    if (@enumToInt(level) <= @enumToInt(std.log.Level.notice)) flush() catch @panic("Can't flush log!");
    lock.release();
}

fn configure_console() void {
    if (std.builtin.os.tag == .windows) {
        // configure windows console - use utf8 and ascii VT100 escape sequences
        const win_con = struct {
            usingnamespace std.os.windows;
            const CP_UTF8: u32 = 65001;
            const ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;
            const ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200;
            //const STD_INPUT_HANDLE: (DWORD) = -10;
            //const STD_OUTPUT_HANDLE: (DWORD) = -11;
            //const STD_ERROR_HANDLE: (DWORD) = -12;
            pub extern "kernel32" fn SetConsoleOutputCP(wCodePageID: std.os.windows.UINT) BOOL;
            pub extern "kernel32" fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) BOOL;
            //pub const GetStdHandle = kernel32.GetStdHandle;
            pub fn configure() void {
                if (SetConsoleOutputCP(CP_UTF8) == 0) {
                    scoped(.console).err("Can't configure windows console to UTF8!", .{});
                }
                var stdout_handle: HANDLE = GetStdHandle(STD_OUTPUT_HANDLE) catch |err| {
                    scoped(.console).err("Windows: Can't get stdout handle! err: {}", .{err});
                    return;
                };
                // var stdin_handle: HANDLE = GetStdHandle(STD_INPUT_HANDLE) catch |err| {
                //     scoped(.console).err("Windows: Can't get stdin handle! err: {}", .{err});
                //     return;
                // };
                var stderr_handle: HANDLE = GetStdHandle(STD_ERROR_HANDLE) catch |err| {
                    scoped(.console).err("Windows: Can't get stderr handle! err: {}", .{err});
                    return;
                };
                // Get console mode
                var stdout_mode: DWORD = 0;
                //var stdin_mode: DWORD = 0;
                var stderr_mode: DWORD = 0;
                if (kernel32.GetConsoleMode(stdout_handle, &stdout_mode) == 0) {
                    scoped(.console).err("Windows can't get stdout console mode! {}", .{kernel32.GetLastError()});
                }
                // if (kernel32.GetConsoleMode(stdin_handle, &stdout_mode) == 0) {
                //     scoped(.console).err("Windows can't get stdin_mode console mode! {}", .{kernel32.GetLastError()});
                // }
                if (kernel32.GetConsoleMode(stderr_handle, &stderr_mode) == 0) {
                    scoped(.console).err("Windows can't get stderr console mode! {}", .{kernel32.GetLastError()});
                }
                // set ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if (SetConsoleMode(stdout_handle, stdout_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                    scoped(.console).err("Windows can't set stdout console mode! {}", .{kernel32.GetLastError()});
                }
                // if (SetConsoleMode(stdin_handle, stdin_mode | ENABLE_VIRTUAL_TERMINAL_INPUT) == 0) {
                //     scoped(.console).err("Windows can't set stdin_mode console mode! {}", .{kernel32.GetLastError()});
                // }
                if (SetConsoleMode(stderr_handle, stderr_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0) {
                    scoped(.console).err("Windows can't set stderr console mode! {}", .{kernel32.GetLastError()});
                }
            }
        };
        _ = win_con;
        win_con.configure();
    }
}

pub fn testOut() void {
    scoped(.testOut).emerg("emerg", .{});
    scoped(.testOut).alert("alert", .{});
    scoped(.testOut).crit("crit", .{});
    scoped(.testOut).err("err", .{});
    scoped(.testOut).warn("warn", .{});
    scoped(.testOut).notice("notice", .{});
    scoped(.testOut).info("info", .{});
    scoped(.testOut).debug("debug", .{});
}
