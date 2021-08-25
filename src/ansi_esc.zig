// ANSI/VT100 escape codes and helper utilities for basic terminal formatting supported by most terminals
// see: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
const std = @import("std");
pub const code = struct {
    // general
    pub const BEL = "\x07";
    pub const BS  = "\x08";
    pub const HT  = "\x09";
    pub const LF  = "\x0A";
    pub const VT  = "\x0B";
    pub const FF  = "\x0C";
    pub const CR  = "\x0D";
    pub const ESC = "\x1B";
    pub const DEL = "\x07";

    // foreground color
    pub const fg_color = struct {
        pub const black     =   "30";
        pub const red       =   "31";
        pub const green     =   "32";
        pub const yellow    =   "33";
        pub const blue      =   "34";
        pub const magenta   =   "35";
        pub const cyan      =   "36";
        pub const white     =   "37";
    };
    // background colors
    pub const bg_color = struct {
        pub const black     = "40";
        pub const red       = "41";
        pub const green     = "42";
        pub const yellow    = "43";
        pub const blue      = "44";
        pub const magenta   = "45";
        pub const cyan      = "46";
        pub const white     = "47";
    };
};

pub const cursor = struct {
    pub fn home() []u8 { return code.ESC ++ "[H"; } //moves cursor to home position (0, 0)
    // should be same as: code.ESC ++ "[{line};{column}f"
    //pub fn goTo(line : i32, column : i32) []u8 { return code.ESC ++ "[{line};{column}H"; }
    pub fn up() []u8 {return code.ESC ++ "[#A"; } //moves cursor up # lines
    pub fn down() []u8 {return code.ESC ++ "[#B"; } //moves cursor down # lines
    pub fn right() []u8 {return code.ESC ++ "[#C"; } //moves cursor right # columns
    pub fn left() []u8 {return code.ESC ++ "[#D"; } //moves cursor left # columns
    pub fn nextL() []u8 {return code.ESC ++ "[#E"; } //moves cursor to beginning of next line, # lines down
    pub fn prevL() []u8 {return code.ESC ++ "[#F"; } //moves cursor to beginning of previous line, # lines up
    //pub fn column() comptime []u8 {return code.ESC ++ "[#G"; } //moves cursor to column #
    pub fn get() []u8 { return code.ESC ++ "[6n"; } //request cursor position (reports as ESC[#;#R)
};

pub const style = struct {
    pub const reset = code.ESC ++ "[0m"; // resets style
    pub const bold = code.ESC ++ "[1m"; // set bold mode.
    pub const dim = code.ESC ++ "[2m"; // set dim/faint mode.
    pub const italic = code.ESC ++ "[3m"; // set italic mode.
    pub const underline = code.ESC ++ "[4m"; // set underline mode.
    pub const blinking = code.ESC ++ "[5m"; // set blinking mode
    pub const inverse = code.ESC ++ "[7m"; // set inverse/reverse mode
    pub const invisible = code.ESC ++ "[8m"; // set invisible mode
    pub const strikethrough = code.ESC ++ "[9m"; // set strikethrough mode.

    // foreground color
    pub const fg = struct {
        pub fn rgb(comptime r : u8, comptime g : u8, comptime b : u8) []u8 {
            return code.ESC ++ "[38;2;" ++ std.fmt.formatInt(r, 10, 0, .{}) ++ ";" ++ std.fmt.formatInt(g, 10, 0, .{}) ++ ";" ++ std.fmt.formatInt(b, 10, 0, .{}) ++ "m";
        }

        // 8 bit colors
        pub const black   = code.ESC++"[" ++ code.fg_color.black ++ "m";
        pub const red     = code.ESC++"[" ++ code.fg_color.red ++ "m";
        pub const green   = code.ESC++"[" ++ code.fg_color.green ++ "m";
        pub const yellow  = code.ESC++"[" ++ code.fg_color.yellow ++ "m";
        pub const blue    = code.ESC++"[" ++ code.fg_color.blue ++ "m";
        pub const magenta = code.ESC++"[" ++ code.fg_color.magenta ++ "m";
        pub const cyan    = code.ESC++"[" ++ code.fg_color.cyan ++ "m";
        pub const white   = code.ESC++"[" ++ code.fg_color.white ++ "m";
    };
    // background color
    pub const bg = struct {
        // 8 bit colors
        pub const black   = code.ESC++"[" ++ code.bg_color.black ++ "m";
        pub const red     = code.ESC++"[" ++ code.bg_color.red ++ "m";
        pub const green   = code.ESC++"[" ++ code.bg_color.green ++ "m";
        pub const yellow  = code.ESC++"[" ++ code.bg_color.yellow ++ "m";
        pub const blue    = code.ESC++"[" ++ code.bg_color.blue ++ "m";
        pub const magenta = code.ESC++"[" ++ code.bg_color.magenta ++ "m";
        pub const cyan    = code.ESC++"[" ++ code.bg_color.cyan ++ "m";
        pub const white   = code.ESC++"[" ++ code.bg_color.white ++ "m";
    };
};

// private
// converts u8 to decimal str like 64 -> "064"
fn u8Str(comptime v : u8) []u8 {
    const val = [3:0]u8 { '0' + (v/100) % 10, '0' + (v/10) % 10, '0' + v % 10 };
    return val;
}