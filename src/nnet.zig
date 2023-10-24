const std = @import("std");
const mem = std.mem;
const math = std.math;
const meta = std.meta;
const trait = std.meta.trait;

pub fn typed(comptime val_t: type) type {
    return struct {
        pub const func = @import("fn_deriv.zig");
        pub const Float = val_t;

        // const LayerType = enum {
        //     Input,
        //     Output,
        //     FeedForward,
        // };

        // pub fn Layer(comptime neurons_: usize, comptime weigts_per_neuron_: usize, comptime activation_: anytype) type {
        //     return struct {
        //         const Self = @This();
        //         pub const neurons_len = neurons_;
        //         pub const weights_per_neuron = weigts_per_neuron_;
        //         pub const bisases_len = biases_; // either 0 or neurons_len
        //         pub const activation = activation_;

        //         out: ?@Vector(neurons_len, Float) = null,
        //         bias: ?@Vector(neurons_len, Float) = null,
        //         weights: ?[neurons]@Vector(weights_per_neuron, Float) = null,

        //         pub fn empty() Self {
        //             return .{ null, null, null };
        //         }
        //     };
        // }

        pub fn TestCase(comptime input_len_: usize, comptime output_len_: usize) type {
            return struct {
                pub const input_len = input_len_;
                pub const output_len = output_len_;
                input: @Vector(input_len, Float),
                answer: @Vector(output_len, Float),
            };
        }

        pub fn TestAccessor(comptime test_case_t: type) type {
            return struct {
                const Self = @This();
                countFn: *const fn (s: *Self) usize,
                grabFn: *const fn (s: *Self, idx: usize) *const test_case_t,
                freeFn: ?*const fn (s: *Self, tc: *const test_case_t) void = null,

                pub fn testCount(self: *Self) usize {
                    return self.countFn(self);
                }

                // call freeTest to release data (for loaders that support it)
                pub fn grabTest(self: *Self, idx: usize) *const test_case_t {
                    return self.grabFn(self, idx);
                }

                pub fn freeTest(self: *Self, tc: *const test_case_t) void {
                    if (self.freeFn) |f| return f(self, tc);
                }
            };
        }

        // Feed-Forward
        // takes current neurons throught weights propogates it forward to next layer neurons `out`, then activates the neurons to `out_activated`
        // `out` can be void - activation will be done (if its not void) however un-activated output won't be stored and will be discarded
        // `out_activated` can be void - activation will be skipped
        pub fn forward(neurons: anytype, weights: anytype, next_activation: anytype, next_biases: anytype, out_activated: anytype) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            // comptime checks
            const do_activate: bool = comptime ablk: {
                if (@TypeOf(out_activated) != @TypeOf(void)) {
                    if (@typeInfo(@TypeOf(out_activated)) != .Pointer)
                        @compileError("output_values has to be writeable pointer!");
                    if (@typeInfo(@TypeOf(out_activated.*)) != .Vector)
                        @compileError("output_values must be pointer to vector!");

                    break :ablk true;
                } else break :ablk false;
            };
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
                res += @as(@Vector(olen, Float), @splat(neurons[nidx])) * weights[nidx];
            }
            if (do_activate) {
                assertFinite(res, "feedforward: res");
                out_activated.* = next_activation.f(res);
                assertFinite(out_activated.*, "feedforward: out_activated");
            }
        }

        // d_oerr_o_na  = 𝝏err_total / 𝝏h  = how much Output error for {layer + 1} changes with respect to output (non activated)
        //
        // pub fn backpropHidden(d_oerr_o_na : anytype, ) void {

        // }

        pub fn randomArray(rnd: std.rand.Random, comptime t: type, comptime len: usize) [len]t {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            var rv: [len]Float = undefined;
            const coef = 1 / @as(Float, len);
            for (rv) |*v| {
                v.* = @as(Float, @floatCast(rnd.floatNorm(f64))) * coef;
            }
            return rv;
        }

        pub fn randomize(rnd: std.rand.Random, out: anytype) void {
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

        pub fn isFinite(v: anytype) bool {
            const ti = @typeInfo(@TypeOf(v));
            if (ti == .Vector) {
                const len = ti.Vector.len;
                var i: usize = 0;
                while (i < len) : (i += 1) {
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

        pub fn assertFinite(v: anytype, msg: []const u8) void {
            if (std.builtin.mode != .Debug and std.builtin.mode != .ReleaseSafe)
                return;
            if (!isFinite(v)) {
                std.debug.panic("Values aren't finite!\n{s}\n{}", .{ msg, v });
            }
        }
    };
}
