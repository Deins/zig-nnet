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
err_mut: std.Thread.Mutex = .{},
err: std.io.BufferedWriter(buff_size, std.fs.File.Writer),

pub fn init() void {
    configure_console();
    log_ctx = .{
        .out = .{ .unbuffered_writer = std.io.getStdOut().writer() },
        .err = .{ .unbuffered_writer = std.io.getStdErr().writer() },
    };
}

pub fn deinit() !void {
    try log_ctx.out.flush();
    try log_ctx.err.flush();
}

pub fn getCtx() *LogCtx { return &log_ctx; }

pub const scoped = std.log.scoped;

pub fn log(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@enumToInt(level) > @enumToInt(log_level)) return;
    //const use_err = @enumToInt(level) > @enumToInt(std.log.Level.warn);
    const use_err = false;
    const flush = if (use_err) log_ctx.out.flush else log_ctx.err.flush;
    var out = if (use_err) log_ctx.out.writer() else log_ctx.err.writer();
    const outm = if (use_err) &log_ctx.out_mut else &log_ctx.err_mut;
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
        .debug => comptime ansi.style.fg.col256(245),
    };
    //const scope_prefix = "(" ++ @tagName(scope) ++ "):\t";
    //const prefix = "[" ++ @tagName(level) ++ "] " ++ scope_prefix;
    const prefix = "";
    const lock = outm.acquire();
    out.print(color ++ prefix ++ format ++ ansi.style.reset ++ "\n", args) catch @panic("Can't write log!");
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
            //const STD_INPUT_HANDLE: (DWORD) = -10;
            //const STD_OUTPUT_HANDLE: (DWORD) = -11;
            //const STD_ERROR_HANDLE: (DWORD) = -12;
            pub extern "kernel32" fn SetConsoleOutputCP(wCodePageID: std.os.windows.UINT) BOOL;
            pub extern "kernel32" fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) BOOL;
            pub fn configure() void {
                if (SetConsoleOutputCP(CP_UTF8) == 0) {
                    scoped(.console).warn("Can't configure windows console to UTF8!", .{});
                }
                //if (GetStdHandle(STD_ERROR_HANDLE)) |h| {
                //    _ = SetConsoleMode(h, ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                //} else |err| std.log.alert("Windows can't configure console: {}", .{err});
                //if (GetStdHandle(STD_OUTPUT_HANDLE)) |h| {
                //    _ = SetConsoleMode(h, ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                //} else |err| std.log.alert("Windows can't configure console: {}", .{err});
            }
        };
        _ = win_con;
        win_con.configure();
    }
}
