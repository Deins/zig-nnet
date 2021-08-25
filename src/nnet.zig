const std = @import("std");
const mem = std.mem;
const math = std.math;
const meta = std.meta;
const trait = std.meta.trait;

pub fn typed(comptime val_type: type) type {
    return struct {
        pub const func = @import("fn_deriv.zig");

        // Feed-Forward
        // takes current neurons throught weights propogates it forward to next layer neurons `out`, then activates the neurons to `out_activated`
        // `out` can be void - activation will be done (if its not void) however un-activated output won't be stored and will be discarded
        // `out_activated` can be void - activation will be skipped 
        pub fn forward(neurons: anytype, weights: anytype, next_activation: anytype, next_biases: anytype, out : anytype, out_activated : anytype) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            // comptime checks
            const do_activate : bool = comptime ablk:{
                if (@TypeOf(out_activated) != @TypeOf(void)) {
                    if (@typeInfo(@TypeOf(out_activated)) != .Pointer)
                        @compileError("output_values has to be writeable pointer!");
                    if (@typeInfo(@TypeOf(out_activated.*)) != .Vector)
                        @compileError("output_values must be pointer to vector!");

                    break :ablk true;
                } else break :ablk false;
            };
            const store_out : bool = comptime if (@TypeOf(out) == @TypeOf(void)) false else true; 
            const olen: u32 = comptime @typeInfo(@TypeOf(weights[0])).Vector.len;
            const nlen: u32 = comptime @typeInfo(@TypeOf(neurons)).Vector.len;
            comptime if (@typeInfo(@TypeOf(neurons)) != .Vector)
                @compileError("neurons must be vector!");
            comptime if (@typeInfo(@TypeOf(weights)) != .Array and @typeInfo(@TypeOf(weights)) != .Pointer and @typeInfo(@TypeOf(weights[0])) != .Vector)
                @compileError("weights have to be array or slice of vectors!");
            comptime if (nlen != weights.len or olen != @typeInfo(@TypeOf(weights[0])).Vector.len)
                @compileError("weights have to be array of [neurons.len] of @vectors(output.len)!");

            // Compute
            var nidx: u32 = 0;
            var res = next_biases;
            while (nidx < nlen) : (nidx += 1) {
                res += @splat(olen, neurons[nidx]) * weights[nidx];
            }
            if (store_out) {
                out.* = res;
            }
            if (do_activate) {
                assertFinite(res, "feedforward: res");
                out_activated.* = next_activation.f(res);
                assertFinite(out_activated.*, "feedforward: out_activated");
            }
        }

        pub fn dot(a : anytype, b : anytype) val_type {
            return @reduce(.Add, a * b); 
        }

        pub fn transpose(inp : anytype, out: anytype) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            if (@typeInfo(@TypeOf(inp)) != .Array) @compileError("input must be array of vectors!");
            if (@typeInfo(@TypeOf(inp[0])) != .Vector) @compileError("input must be array of vectors!");
            if (@typeInfo(@TypeOf(out)) != .Pointer) @compileError("output must be pointer to array of vectors!");
            const n = inp.len;
            const m = @typeInfo(@TypeOf(inp[0])).Vector.len;
            if (@typeInfo(@TypeOf(out.*)).Array.len != m or @typeInfo(@TypeOf(out.*[0])).Vector.len != n) {
                @compileError("input NxM has to have output MxN for transpose!");
            }
            var res : @TypeOf(out.*) = undefined;
            var in : usize = 0;
            var im : usize = 0;

            // TODO: optimize - this might help:
            //      https://fgiesen.wordpress.com/2013/07/09/simd-transposes-1/
            //      https://fgiesen.wordpress.com/2013/08/29/simd-transposes-2/

            // comptime fn genMasks (comptime n : usize, comptime m : usize) [m]@Vector( n, val_type) {
            //     var masks : [m]@Vector( n, val_type) = undefined;
            //     var i : usize = 0;
            //     while (i < n) : (i+=1) {
            //         masks[i] = 
            //     }
            // };

            while (in < n) : (in += 1){
                while (im < m) : (im += 1) {
                    res[im][in] = inp[in][im];
                }
            }
            out.* = res;
        }

        pub fn mat_mult(a : anytype, b : anytype, out : anytype) void {
            // naming assumes column-major
            const a_cols = a.len;
            const a_rows = std.mem.len(a[0]);
            const b_cols = b.len;
            const b_rows = std.mem.len(b[0]);

            comptime if (a_cols != b_rows) 
                @compileError("Matrix multiplication requires that columns in A equals the number of rows in B!");
            comptime if (@typeInfo(@TypeOf(out)) != .Pointer)
                @compileError("Output must be pointer!");
            comptime if (std.meta.child(out).len != b_cols and std.meta.child(out).child.Vector.len != a_rows)
                @compileError("Output has invalid size!");
            
            // TODO: optimize, this might help: https://malithjayaweera.com/2020/07/blocked-matrix-multiplication/
            var at : @TypeOf(a) = undefined;
            transpose(a, &tmp);
            var i : usize  = 0;
            while (i < b_cols) : (i+=1) {
                var j : usize  = 0;
                while (j < a_rows) : (j+=1) {
                    out[i][j] = dot(at[i], b[i]);
                }
            }
        }

        pub fn randomArray(rnd: *std.rand.Random, comptime t: type, comptime len: usize) [len]t {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            var rv: [len]val_type = undefined;
            const coef = 1 / @as(val_type, len);
            for (rv) |*v| {
                v.* = @floatCast(val_type, rnd.floatNorm(f64)) * coef;
            }
            return rv;
        }

        pub fn randomize(rnd: *std.rand.Random, out: anytype) void {
            comptime if (@typeInfo(@TypeOf(out)) != .Pointer)
                @compileError("out must be pointer!");
            const tinfo = @typeInfo(@TypeOf(out.*));
            if (tinfo == .Vector) {
                const arr = randomArray(rnd, tinfo.Vector.child, tinfo.Vector.len);
                out.* = arr;
            } else if (tinfo == .Array) {
                const cinfo = @typeInfo(tinfo.Array.child);
                if (cinfo == .Array or cinfo == .Vector) {
                    for (out.*) |*o| {
                        randomize(rnd, o);
                    }
                } else {
                    out.* = randomArray(rnd, tinfo.Vector.child, tinfo.Vector.len);
                    @compileError("unexpected!");
                }
            }
        }

        pub fn isFinite(v : anytype) bool{
            const ti = @typeInfo(@TypeOf(v));
            if (ti == .Vector) {
                const len = ti.Vector.len;
                var i : usize = 0;
                while (i < len) : (i+=1){
                    if (!std.math.isFinite(v[i])) 
                        return false;
                }
            } else {
                for (v) |vi| {
                    if (!std.math.isFinite(vi)) 
                        return false;
                }
            }
            return true;
        }

        pub fn assertFinite(v : anytype, msg : []const u8) void {
            if (std.builtin.mode != .Debug and std.builtin.mode != .ReleaseSafe)
                return;
            if (!isFinite(v)) {
                std.debug.panic("Values aren't finite!\n{s}\n{}", .{msg, v});
            }
        }
    };
}
