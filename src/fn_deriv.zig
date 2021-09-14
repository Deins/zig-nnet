// Activation and other functions
// each function is encapsulated in struct that can contain:
//  * function istelf:                  f(x)
//  * derivative:                       deriv(x)
//  * derivative from input z = f(x):   derivZ(z)
//  * derive against other input:       derivZY(z, y)
// most functions support both scalars and @Vector with exceptions such as softmax

const std = @import("std");
const math = std.math;
const meta = std.meta;
const Optimized = std.builtin.FloatMode.Optimized;

// PRIVATE UTILITIES:

// helper function for constants to work with both vectors and scalars
inline fn splat(comptime t: type, val: anytype) t {
    @setFloatMode(Optimized);
    comptime if (@typeInfo(t) == .Vector) {
        return @splat(@typeInfo(t).Vector.len, @as(@typeInfo(t).Vector.child, val));
    } else {
        return val;
    };
}

// WARNING: might not be applicable to all functions
// d/dx σ(x) = σ(x) * (1 − σ(x))
// z = σ(x)
fn usualDerivZ(z: anytype) @TypeOf(z) {
    @setFloatMode(Optimized);
    const t = @TypeOf(z);
    return z * (splat(t, 1) - z);
}

// PUBLIC UTILITIES:

// // chooses available derivation function if both x and y is known
// inline pub deriv_xy(fn_struct : type, x : anytype, y : anytype) @TypeOf(x) {
//     const ti = @typeInfo(fn_struct);
//     @compileError("todo: implement!");
// }

// FUNCTIONS:

// binary
pub const bin = struct {
    pub fn f(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        return math.clamped(x, 0, 1);
    }
    pub fn deriv(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        @compileError("This function is not continuously differentiable!");
    }
};

pub const sigmoid = struct {
    // σ(x) = 1 / (1 + e^−x)
    pub fn f(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        return splat(t, 1) / (splat(t, 1) + @exp(-x));
    }
    // (e^x) / (e^x + 1)^2
    pub fn deriv(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        const exp = @exp(-x);
        const exp_p1 = (splat(t, 1) + exp);
        return exp / (exp_p1 * exp_p1);
    }
    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x/1225116#1225116
    // d/dx σ(x) = σ(x) * (1 − σ(x))
    // z = σ(x)
    pub const derivZ = usualDerivZ;
};

pub const softmax = struct {
    const Self = @This();
    // Unstable:
    // σ(x) = e^x / Σ(e^x)
    // pub fn f(x: anytype) @TypeOf(x) {
    //     @setFloatMode(Optimized);
    //     const t = @TypeOf(x);
    //     comptime if (@typeInfo(t) != .Vector) @compileError("softmax accepts only Vectors");
    //     const exp = @exp(x);
    //     const exp_sum = @reduce(.Add, exp);
    //     return exp / splat(t, exp_sum);
    // }

    pub fn f(x: anytype) @TypeOf(x) {
        const t = @TypeOf(x);
        const e = @exp(x - splat(t, @reduce(.Max, x)));
        return e / splat(t, @reduce(.Add, e));
    }

    // from: https://themaverickmeerkat.com/2019-10-23-Softmax/
    // 𝝏σ(x) / 𝝏x = -σ(x) * σ(y)
    // z = σ(x)
    pub const derivZ = usualDerivZ;

    // from: https://themaverickmeerkat.com/2019-10-23-Softmax/
    // 𝝏σ(x) / 𝝏y = -σ(x) * σ(y)
    // z = σ(x)
    pub fn derivZY(z: anytype, y: anytype) @TypeOf(z) {
        @setFloatMode(Optimized);
        return -z * Self.f(y);
    }
};

pub const relu = struct {
    pub fn f(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        return @maximum(splat(@TypeOf(x), 0), x);
    }
    // =1 when >0 or 0 otherwise
    pub fn deriv(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        return @maximum(splat(t, 0), @minimum(splat(t, 1), @ceil(x)));
    }

    pub const derivZ = deriv; // special case for relu - not applicable to other fn
};

pub const relu_leaky = struct {
    const coef = 0.1;
    pub fn f(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        return @maximum(x, x * splat(@TypeOf(x), coef));
    }
    // =1 when >0 or 0 otherwise
    pub fn deriv(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        return @maximum(splat(t, coef), @minimum(splat(t, 1), @ceil(x)));
    }

    pub const derivZ = deriv; // special case for relu - not applicable to other fn
};

pub const relu6 = struct {
    const max = 6;
    pub fn f(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        return @maximum(splat(t, 0), @minimum(x, splat(t, max)));
    }

    pub fn deriv(x: anytype) @TypeOf(x) {
        @setFloatMode(Optimized);
        const t = @TypeOf(x);
        const div = 1.0 / @as(comptime_float, max);
        const nc = @ceil(splat(t, div) * x);
        const cut = @minimum(splat(t, 1), nc) - @maximum(splat(t, 0), nc);
        return @maximum(splat(t, 0), cut);
    }

    pub const derivZ = deriv; // special case for relu - not applicable to other fn
};

pub const none = struct {
    pub fn f(x: anytype) @TypeOf(x) {
        return x;
    }
    pub fn deriv(x: anytype) @TypeOf(x) {
        return splat(@TypeOf(z), 1);
    }
    pub fn derivZ(z: anytype) @TypeOf(z) {
        return splat(@TypeOf(z), 1);
    }
};

// Error funcs

pub const absErr = struct {
    // answer = correct answer, predicted = output from nnet
    pub fn f(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        return (answer - predicted);
    }

    pub fn deriv(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        return (answer - predicted);
    }
};

pub const squaredErr = struct {
    // answer = correct answer, predicted = output from nnet
    pub fn f(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        return  (answer - predicted) * (answer - predicted);
    }

    pub fn deriv(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        return splat(@TypeOf(answer), 2) * (answer - predicted);
    }
};

pub const logLoss = struct {
    // answer = correct answer, predicted = output from nnet
    pub fn f(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        const t = @TypeOf(answer);
        const ti = @typeInfo(t);
        const p = @minimum(@maximum(predicted, splat(t, 0.00001)), splat(t, 0.99999));
        const one = @log(p) * answer;
        const zero = @log(splat(t, 1) - p) * (splat(t, 1) - answer);
        const avg = splat(t, 1.0 / @intToFloat(ti.Vector.child, ti.Vector.len));
        return (one + zero) * -avg;
    }

    pub fn deriv(answer: anytype, predicted: anytype) @TypeOf(answer) {
        @setFloatMode(Optimized);
        const t = @TypeOf(answer);
        const ti = @typeInfo(t);
        const p = @minimum(@maximum(predicted, splat(t, 0.00001)), splat(t, 0.99999));
        const one = (splat(t, 1) / p) * answer;
        const zero = (splat(t, -1) / (splat(t, 1) - p)) * (splat(t, 1) - answer);
        const avg = splat(t, 1.0 / @intToFloat(ti.Vector.child, ti.Vector.len));
        return (one + zero) * avg;
    }

    //pub const derivZ = usualDerivZ;
};
