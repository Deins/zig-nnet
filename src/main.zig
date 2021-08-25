const std = @import("std");
const mem = std.mem;
const os = std.os;
const debug = std.debug;
const io = std.io;
const json = std.json;
const fs = std.fs;
const fmtDuration = std.fmt.fmtDuration;
const bin_file = @import("bin_file.zig");

const Float = f32;
const ansi = @import("ansi_esc.zig");
const tdata = @import("tdata.zig").forData(f32, 28 * 28, 10);
const TrainingData = tdata.TrainingData;
const TestCase = tdata.TestCase;
const nnet = @import("nnet.zig").typed(Float);

const Options = struct {
    workers: usize = 0,
    load: ?[]const u8 = null,
    save: ?[]const u8 = null,
    epoches: usize = 1,
    learn_rate: Float = 0.1,
    batch_size: usize = 16,
};
var options: Options = .{};

const rlog = std.log.scoped(.raw);

const LogCtx = struct {
    const Self = @This();
    const buff_size = 1024;
    out_mut: std.Thread.Mutex = .{},
    out: std.io.BufferedWriter(buff_size, std.fs.File.Writer),
    err_mut: std.Thread.Mutex = .{},
    err: std.io.BufferedWriter(buff_size, std.fs.File.Writer),

    pub fn init() Self {
        return .{
            .out = .{ .unbuffered_writer = std.io.getStdOut().writer() },
            .err = .{ .unbuffered_writer = std.io.getStdErr().writer() },
        };
    }

    pub fn configure_console() void {
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
                        debug.warn("Can't configure windows console to UTF8!", .{});
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
};
var log_ctx: LogCtx = undefined;

pub const log_level: std.log.Level = std.log.Level.info;
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

    const scope_prefix = "(" ++ @tagName(scope) ++ "): ";
    const prefix = "[" ++ @tagName(level) ++ "] " ++ scope_prefix;
    const lock = outm.acquire();
    out.print(prefix ++ format ++ "\n", args) catch @panic("Can't write log!");
    if (@enumToInt(level) <= @enumToInt(std.log.Level.notice)) flush() catch @panic("Can't flush log!");
    lock.release();
}

const NNet = struct {
    const Self = @This();
    pub const ValType = Float;

    // neural net layer sizes
    //const sizes = [4]usize{ 28 * 28, 128, 64, 10 };
    pub const sizes = [4]usize{ 28 * 28, 28 * 28, 64, 10 };
    pub const input_len = sizes[0];
    pub const output_len = sizes[sizes.len - 1];

    // activation functions
    pub const a1 = nnet.func.sigmoid;
    pub const a2 = nnet.func.sigmoid;
    pub const a3 = nnet.func.sigmoid;

    pub const err_fn = nnet.func.squaredErr;

    pub const TrainResult = struct {
        correct: u32 = 0, // was answer correct, in case of batch: how many?
        loss: Float = std.math.nan(Float),
        // for merges: how may tests this struct represent
        //  after merge is finalized this is inversed to negative number to prevent accidental wrong merges etc.
        test_cases: Float = 0,
        // Backprop weight derivatives
        d_w0: [sizes[0]]@Vector(sizes[1], Float) = undefined,
        d_w1: [sizes[1]]@Vector(sizes[2], Float) = undefined,
        d_w2: [sizes[2]]@Vector(sizes[3], Float) = undefined,

        // merges training results for batch training
        //  derivatives are summed, call finalizeMerge() before applying to average them
        pub fn merge(self: *TrainResult, b: TrainResult) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            if (self.test_cases == 0) {
                self.* = b;
                return;
            }
            std.debug.assert(self.test_cases >= 1);
            std.debug.assert(b.test_cases >= 1);
            self.correct += b.correct;
            self.loss += b.loss;
            self.test_cases += b.test_cases;

            for (self.d_w0) |*w, nidx| {
                w.* += b.d_w0[nidx];
            }

            for (self.d_w1) |*w, nidx| {
                w.* += b.d_w1[nidx];
            }

            for (self.d_w2) |*w, nidx| {
                w.* += b.d_w2[nidx];
            }
        }

        pub fn finalizeMerge(self: *TrainResult) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            std.debug.assert(self.test_cases >= 1);
            if (self.test_cases == 1) return;
            const n: Float = 1.0 / self.test_cases;
            if (!std.math.isFinite(n)) debug.panic("Not finite: {} / {} = {}", .{ 1.0, self.test_cases, n });
            self.loss *= n;
            for (self.d_w0) |*w| {
                w.* *= @splat(@typeInfo(@TypeOf(w.*)).Vector.len, n);
            }

            for (self.d_w1) |*w| {
                w.* *= @splat(@typeInfo(@TypeOf(w.*)).Vector.len, n);
            }

            for (self.d_w2) |*w| {
                w.* *= @splat(@typeInfo(@TypeOf(w.*)).Vector.len, n);
            }
            self.test_cases *= -1;
        }
    };

    // Member variables:
    //  variables that contain index, its from 0 .., where 0 = input layer, and 1 is first hidden layer etc.

    // Neurons:
    // input
    i: @Vector(sizes[0], Float) = undefined,
    // hidden layers
    h1: @Vector(sizes[1], Float) = undefined,
    h2: @Vector(sizes[2], Float) = undefined,
    // output non activated and activated
    //out: @Vector(sizes[3], Float) = undefined,
    out_activated: @Vector(sizes[3], Float) = undefined,

    // Biases
    b1: @Vector(sizes[1], Float) = undefined,
    b2: @Vector(sizes[2], Float) = undefined,
    bo: @Vector(sizes[3], Float) = undefined,

    // Weights
    w0: [sizes[0]]@Vector(sizes[1], Float) = undefined,
    w1: [sizes[1]]@Vector(sizes[2], Float) = undefined,
    w2: [sizes[2]]@Vector(sizes[3], Float) = undefined,

    pub fn randomize(self: *Self, rnd: *std.rand.Random) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        nnet.randomize(rnd, &self.w0);
        nnet.randomize(rnd, &self.b1);
        nnet.randomize(rnd, &self.w1);
        nnet.randomize(rnd, &self.b2);
        nnet.randomize(rnd, &self.w2);
        nnet.randomize(rnd, &self.bo);
        //nnet.randomize(rnd, &self.b4);
        self.bo = @splat(@typeInfo(@TypeOf(self.bo)).Vector.len, @as(Float, 0));
    }

    // train to get derivatives
    pub fn trainDeriv(self: *Self, test_case: TestCase, train_result: *TrainResult) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        //var timer = try std.time.Timer.start();
        debug.assert(std.mem.len(test_case.input) == std.mem.len(self.i));
        self.i = test_case.input;
        self.h1 = self.i;
        // nnet.forward(self.i, self.w0, Self.a1, self.b1, void, &self.h1);
        nnet.forward(self.h1, self.w1, Self.a2, self.b2, void, &self.h2);
        nnet.forward(self.h2, self.w2, Self.a3, self.bo, void, &self.out_activated);
        nnet.assertFinite(self.out_activated, "out_activated");

        const predicted_confidence: Float = @reduce(.Max, self.out_activated);
        var answer_vector = test_case.answer;
        const answer: u8 = ablk: {
            var i: u8 = 0;
            while (i < output_len) : (i += 1) {
                if (answer_vector[i] == 1) {
                    break :ablk i;
                }
            }
            @panic("Wrongly formated answer!");
        };
        const o_err = err_fn.f(answer_vector, self.out_activated);
        const total_err: Float = @reduce(.Add, o_err);
        const predicted_correct = predicted_confidence == self.out_activated[answer] and predicted_confidence > 0.2 and total_err < 0.5;

        // BACKPROP:
        //  from: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        // Last/Output layer:
        // how much total error change with respect to the activated output:
        //      𝝏err_total / 𝝏out_activated
        const d_err_oa = -err_fn.deriv(answer_vector, self.out_activated);
        nnet.assertFinite(d_err_oa, "backprop out: d_err_oa");
        // how much activated output change with respect to the (non activated) output:
        //      𝝏out_activated / 𝝏out
        const d_oa_o = a3.derivZ(self.out_activated);
        nnet.assertFinite(d_oa_o, "backprop out: d_oa_o");
        // iterate last hidden layer neurons and update its weights
        for (self.w2) |_, nidx| {
            // how much (non activated) output change with respect to the weights:
            //      𝝏out / 𝝏w
            const d_o_w = self.h2; // last hidden layer
            // how much total error change with respect to the weights:
            //      𝝏total_err / 𝝏w
            const d_err_w = d_err_oa * d_oa_o * @splat(sizes[3], d_o_w[nidx]);
            nnet.assertFinite(d_err_w, "backprop out: d_err_w");
            train_result.d_w2[nidx] = d_err_w; // store result
        }
        // how much Output error changes with respect to output (non activated):
        //      𝝏err_o / 𝝏h2_na
        const d_oerr_o_na = d_err_oa * d_oa_o;

        { // Hidden layer
            const h_len = @typeInfo(@TypeOf(self.h2)).Vector.len;
            // how much total error changes with respect to output (activated) of hidden layer
            //      𝝏err_total / 𝝏h
            var d_err_h: @Vector(h_len, Float) = undefined;
            for (self.w2) |w, nidx| {
                // how much error of output (not activated) changes with respect to hidden layer output (activated)
                //      𝝏err_o_na / 𝝏err_h
                const d_err_o_na__err_h = d_oerr_o_na * w;
                // 𝝏err_total / 𝝏h
                d_err_h[nidx] = @reduce(.Add, d_err_o_na__err_h);
                nnet.assertFinite(d_err_h, "backprop hidden out->current: d_err_h");
            }
            // how much output of hidden_activated changes with respect to hidden non activated
            //      𝝏h2 / 𝝏h2_na
            const d_h2_h2na = a1.derivZ(self.h2);
            nnet.assertFinite(d_h2_h2na, "backprop a1.derivZ( d_h2_h2na )");

            for (self.w1) |_, nidx| {
                // how much hidden layer (non activated) output change with respect to the weights:
                //      𝝏out / 𝝏w
                const d_o_w = self.h1;
                // how much total error change with respect to the weights:
                //      𝝏total_err / 𝝏w
                const d_err_w = d_err_h * d_h2_h2na * @splat(h_len, d_o_w[nidx]);
                nnet.assertFinite(d_err_w, "backprop hidden: d_err_w");
                // store result
                train_result.d_w1[nidx] = d_err_w;
            }
        }
        train_result.correct = if (predicted_correct) 1 else 0;
        train_result.loss = total_err;
        train_result.test_cases = 1;

        if (false) { // Log
            //debug.print("feedForward {}#\t{}\n", .{ ti, fmtDuration(timer.lap()) });
            //debug.print("'Answer: {c}\tTotalErr: {d:.4}'\n", .{ '0' + answer, total_err });

            { // print output
                // ✓✔ ⦸🅧⭙⦸⨂◉⛒⊘⛝✗✘ ➊⓵①
                const symbol = if (predicted_correct) (ansi.style.fg.green ++ "✔") else (ansi.style.fg.red ++ "✘");
                rlog.info("{s}" ++ ansi.style.reset, .{symbol});
                rlog.info("  Answer: {c}\tconfidence: {d:.1}%\tloss: {d:.4}\t", .{ '0' + answer, predicted_confidence * 100, total_err });
                if (!predicted_correct and false) {
                    rlog.info("\n", .{});
                    var i: u8 = 0;
                    while (i < 10) : (i += 1) {
                        rlog.info("'{c}' => {d:.0}%\n", .{ i + '0', self.out_activated[i] * 100 });
                    }
                    rlog.info("=======================\n", .{});
                }
            }
        }

        // std.log.info("Test (Thread #{}) finished in {}. err: {}\n", .{ std.Thread.getCurrentId(), fmtDuration(timer.lap()), dataset_err });
    }

    pub fn learn(self: *Self, train_results: TrainResult, learn_rate: Float) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        for (self.w1) |*w, nidx| {
            w.* -= train_results.d_w1[nidx] * @splat(@typeInfo(@TypeOf(w.*)).Vector.len, @as(Float, learn_rate));
        }
        for (self.w2) |*w, nidx| {
            w.* -= train_results.d_w2[nidx] * @splat(@typeInfo(@TypeOf(w.*)).Vector.len, @as(Float, learn_rate));
        }
    }
};

// const TrainData = struct {
//     inputs = []*[img_width * img_height]Float,
//     input_names: [][]u8,
//     batch_size : usize = 64, // 0 = all data (gradient descent, 1 = Stochastic, 2... = MiniBatch
//     shuffle : ?std.rand.Random // when set option is set - shuffles before each batch
// };

pub fn train(alloc: *mem.Allocator) !void {
    var td = TrainingData.init(alloc);
    try td.load(false);
    const seed = 364123;
    var rnd = std.rand.DefaultPrng.init(seed);
    const Trainer = @import("nnet_trainer.zig").forNet(NNet);
    var trainer = Trainer.init(alloc, &rnd.random);
    trainer.batch_size = options.batch_size;
    trainer.workers = options.workers;
    trainer.learn_rate = options.learn_rate;

    var net: NNet = undefined;
    if (options.load) |p| {
        var in_file = std.fs.cwd().openFile(p, .{}) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
        defer in_file.close();
        bin_file.readFile(NNet, &net, &in_file) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
    } else net.randomize(&rnd.random);

    var timer = try std.time.Timer.start();
    var epoches: usize = options.epoches;
    while (epoches > 0) : (epoches -= 1) {
        try trainer.trainEpoch(&net, &td.accessor);
    }
    debug.print("\nTotal train time: {}\n", .{fmtDuration(timer.lap())});

    if (options.save) |p| {
        var in_file = std.fs.cwd().createFile(p, .{}) catch |err| debug.panic("Can't open file for storing nnet: '{s}' Error:{}", .{ p, err });
        defer in_file.close();
        bin_file.writeFile(NNet, &net, &in_file) catch |err| debug.panic("Can't write nnet to file: '{s}' Error:{}", .{ p, err });
    }

    // var total_err: f64 = 0.0;
    // const Thread = struct {
    //     handle: std.Thread = undefined,
    //     net: NNet,
    // };
    // var threads: []Thread = try alloc.alloc(Thread, cores);
    // defer alloc.free(threads);
    // var train_iters: usize = 1;
    // while (train_iters > 0) : (train_iters -= 1) {
    //     total_err = 0;
    //     var prev_test: usize = 0;
    //     for (threads) |*worker, tidx| {
    //         worker.net = initial_net;
    //         const slice_end = if (tidx + 1 == cores) td.images.items.len else prev_test + td.images.items.len / cores;

    //         worker.handle = try std.Thread.spawn(.{}, NNet.train, .{ &worker.net, td.images.items[prev_test..slice_end], td.answers.items[prev_test..slice_end] });
    //         prev_test = slice_end;
    //     }
    //     // join
    //     debug.print("{} threads launched, waiting for them to finish.\n", .{cores});
    //     for (threads) |*worker| {
    //         std.Thread.join(worker.handle);
    //         //initial_net = worker.net;
    //     }
    //     debug.print("\nTotal train time: {} Total err: {d:.4}\n", .{ fmtDuration(timer.lap()), total_err });
    // }
}

pub fn main() !void {
    LogCtx.configure_console();
    log_ctx = LogCtx.init();
    // rlog.alert("alert", .{});
    // rlog.crit("crit", .{});
    // rlog.debug("debug", .{});
    // rlog.emerg("emerg", .{});
    // rlog.err("err", .{});
    // rlog.info("info", .{});
    // rlog.notice("notice", .{});
    // rlog.warn("warn", .{});

    options.workers = try std.Thread.getCpuCount();
    var galloc = std.heap.GeneralPurposeAllocator(.{}){};
    defer if (galloc.deinit()) {
        debug.panic("GeneralPurposeAllocator had leaks!", .{});
    };
    var arena_log_alloc = std.heap.LoggingAllocator(std.log.Level.debug, std.log.Level.notice).init(&galloc.allocator);
    var arena = std.heap.ArenaAllocator.init(&arena_log_alloc.allocator);
    //var arena = std.heap.ArenaAllocator.init(&galloc.allocator);
    defer arena.deinit();

    { // ARGS
        const args = (try std.process.argsAlloc(&galloc.allocator))[1..]; // skip first arg as it points to current executable
        defer std.process.argsFree(&galloc.allocator, args);
        const printHelp = struct {
            pub fn print() void {
                var cwd_path_buff: [512]u8 = undefined;
                const cwd_path: []const u8 = fs.cwd().realpath("", cwd_path_buff[0..]) catch "<ERROR>";
                debug.print("cwd = {s}\n", .{cwd_path});
                debug.print("Commands:\n", .{});
                debug.print("\tpreprocess\t - saves batch of all input images in single file for faster loading\n", .{});
                debug.print("\ttrain\t- trains network, optional arguments before command:\n", .{});
                debug.print("\t\t--load {{relative_path - instead of initialising random net, load existing}}\n", .{});
                debug.print("\t\t--save {{relative_path - after training save net}}\n", .{});
                debug.print("\t\t--learn-rate {{float}}\n", .{});
                debug.print("\t\t--batch-size {{int}}\n", .{});
                debug.print("\t\t--epoches {{how many epoches to train}}\n", .{});
                debug.print("\t\t--workers {{path - after training save net}}\n", .{});
            }
        }.print;

        if (args.len < 1)
            printHelp();

        // commands
        var skip: i32 = 0;
        for (args) |argv, ai| {
            if (skip > 0) {
                skip -= 1;
                continue;
            }
            if (std.cstr.cmp(argv, "--workers") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.workers = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.cstr.cmp(argv, "--epoches") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.epoches = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.cstr.cmp(argv, "--learn-rate") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.learn_rate = try std.fmt.parseFloat(Float, args[ai + 1]);
                skip = 1;
            } else if (std.cstr.cmp(argv, "--load") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.load = args[ai + 1];
                skip = 1;
            } else if (std.cstr.cmp(argv, "--save") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.save = args[ai + 1];
                skip = 1;
            } else if (std.cstr.cmp(argv, "--batch-size") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.batch_size = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.cstr.cmp(argv, "--epoches") == 0) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.epoches = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.cstr.cmp(argv, "preprocess") == 0) {
                var td = TrainingData.init(&arena.allocator);
                td.load(true) catch |err| debug.panic("Error: {}", .{err});
                td.saveImagesBatch() catch |err| debug.panic("Error: {}", .{err});
            } else if (std.cstr.cmp(argv, "train") == 0) {
                train(&arena.allocator) catch |err| debug.panic("Error: {}", .{err});
            } else if (std.cstr.cmp(argv, "help") == 0) {
                printHelp();
            } else std.debug.panic("Unknown argument: {s}", .{argv});
        }
    }
}
