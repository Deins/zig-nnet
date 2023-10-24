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

const Dataset = @import("csv_img_dataset.zig").forData(
    Float,
    [2]usize{ 28, 28 },
    @as(comptime_int, 10),
);
const TestCase = Dataset.TestCase;
const nnet = @import("nnet.zig").typed(Float);
const LogCtx = @import("log.zig");

const Options = struct {
    workers: usize = 0,
    load: ?[]const u8 = null,
    save: ?[]const u8 = null,
    img_dir_out: ?[]const u8 = null,
    epoches: usize = 1,
    learn_rate: Float = 0.1,
    batch_size: usize = 16,
};
var options: Options = .{};

const rlog = LogCtx.scoped(.raw);
const mlog = LogCtx.scoped(.main);

// setup logging system
pub const log_level = LogCtx.log_level;
pub const log = LogCtx.log;

const NNet2 = struct {
    li: nnet.Layer(28 * 28, 64) = .{},
    l1: nnet.Layer(64, 10) = .{},
    lo: nnet.Layer(10, 0) = .{},
};

const NNet = struct {
    const Self = @This();
    pub const ValType = Float;

    // neural net layer sizes
    //const sizes = [4]usize{ 28 * 28, 128, 64, 10 };
    // TODO: layer 1 is skipped right now
    pub const sizes = [4]usize{ 28 * 28, 28 * 28, 256, 10 };
    pub const input_len = sizes[0];
    pub const output_len = sizes[sizes.len - 1];

    // activation functions
    pub const a1 = nnet.func.relu_leaky;
    pub const a2 = nnet.func.relu_leaky;
    pub const a3 = nnet.func.softmax;

    pub const err_fn = nnet.func.logLoss;

    pub const TrainResult = struct {
        correct: u32 = 0, // was answer correct, in case of batch: how many?
        loss: Float = 0,
        // for merges: how may tests this struct represent
        //  after merge is finalized this is inversed to negative number to prevent accidental wrong merges etc.
        test_cases: Float = 0,
        // Backprop bias derivatives
        d_b1: @Vector(sizes[1], Float) = std.mem.zeroes(@Vector(sizes[1], Float)),
        d_b2: @Vector(sizes[2], Float) = std.mem.zeroes(@Vector(sizes[2], Float)),
        d_bo: @Vector(sizes[3], Float) = std.mem.zeroes(@Vector(sizes[3], Float)),
        // Backprop weight derivatives
        d_w0: [sizes[0]]@Vector(sizes[1], Float) = [_]@Vector(sizes[1], Float){@splat(0)} ** sizes[0],
        d_w1: [sizes[1]]@Vector(sizes[2], Float) = [_]@Vector(sizes[2], Float){@splat(0)} ** sizes[1],
        d_w2: [sizes[2]]@Vector(sizes[3], Float) = [_]@Vector(sizes[3], Float){@splat(0)} ** sizes[2],

        // merges training results for batch training
        //  derivatives are summed, call finalizeMerge() before applying to average them
        pub fn merge(self: *TrainResult, b: TrainResult) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            if (b.test_cases == 0) return;
            if (self.test_cases == 0) {
                self.* = b;
                return;
            }
            std.debug.assert(self.test_cases >= 1);
            std.debug.assert(b.test_cases >= 1);
            self.correct += b.correct;
            self.loss += b.loss;
            self.test_cases += b.test_cases;

            for (self.d_w0, 0..) |*w, nidx| {
                w.* += b.d_w0[nidx];
            }

            for (self.d_w1, 0..) |*w, nidx| {
                w.* += b.d_w1[nidx];
            }

            for (self.d_w2, 0..) |*w, nidx| {
                w.* += b.d_w2[nidx];
            }
        }

        pub fn average(self: *TrainResult) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            std.debug.assert(self.test_cases >= 1);
            if (self.test_cases == 1) return;
            const n: Float = 1.0 / self.test_cases;
            if (!std.math.isFinite(n)) debug.panic("Not finite: {} / {} = {}", .{ 1.0, self.test_cases, n });
            self.loss *= n;
            for (self.d_w0) |*w| {
                w.* *= @splat(n);
            }

            for (self.d_w1) |*w| {
                w.* *= @splat(n);
            }

            for (self.d_w2) |*w| {
                w.* *= @splat(n);
            }
            self.test_cases *= -1;
        }

        pub fn print(self: *@This()) void {
            mlog.info("b1: {d:.2}", .{self.d_b1});
            mlog.info("b2: {d:.2}", .{self.d_b2});
            mlog.info("bo: {d:.2}", .{self.d_bo});
            mlog.info("w0: {d:.2}", .{self.d_w0});
            mlog.info("w1: {d:.2}", .{self.d_w1});
            mlog.info("w2: {d:.2}", .{self.d_w2});
        }
    };

    // Member variables:
    //  variables that contain index, its from 0 .., where 0 = input layer, and 1 is first hidden layer etc.

    // Neurons:
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

    pub fn randomize(self: *Self, rnd: std.rand.Random) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        nnet.randomize(rnd, &self.w0);
        nnet.randomize(rnd, &self.w1);
        nnet.randomize(rnd, &self.w2);
        // nnet.randomize(rnd, &self.b1);
        // nnet.randomize(rnd, &self.b2);
        // nnet.randomize(rnd, &self.bo);
        self.b1 = @splat(0.1);
        self.b2 = @splat(0.1);
        self.bo = @splat(0.1);
        //nnet.randomize(rnd, &self.b4);
        //self.bo = std.mem.zeroes(@TypeOf(self.bo));

        nnet.assertFinite(self.b1, "b1");
        nnet.assertFinite(self.b2, "b2");
        nnet.assertFinite(self.bo, "b3");
    }

    pub fn feedForward(self: *Self, input: *const @Vector(sizes[0], Float)) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        self.h1 = input.*;
        // nnet.forward(self.i, self.w0, Self.a1, self.b1, void, &self.h1);
        nnet.forward(self.h1, self.w1, Self.a2, self.b2, &self.h2);
        nnet.forward(self.h2, self.w2, Self.a3, self.bo, &self.out_activated);
        nnet.assertFinite(self.out_activated, "out_activated");
    }

    // train to get derivatives
    pub fn trainDeriv(self: *Self, test_case: TestCase, train_result: *TrainResult) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        //var timer = try std.time.Timer.start();
        debug.assert(std.mem.len(test_case.input) == sizes[0]);
        self.feedForward(&test_case.input);

        const predicted_confidence: Float = @reduce(.Max, self.out_activated);
        //mlog.debug("output layer: {}", .{self.out_activated});
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
        // how much Output error changes with respect to output (non activated):
        //      𝝏err_o / 𝝏h2_na
        const d_oerr_o_na = d_err_oa * d_oa_o;
        {
            // how much (non activated) output change with respect to the weights:
            //      𝝏out / 𝝏w
            const d_o_w = self.h2; // last hidden layer
            train_result.d_bo += d_oerr_o_na; // store result
            // iterate last hidden layer neurons and update its weights
            for (self.w2, 0..) |_, nidx| {
                // how much total error change with respect to the weights:
                //      𝝏total_err / 𝝏w
                const d_err_w = d_oerr_o_na * @as(@Vector(sizes[3], Float), @splat(d_o_w[nidx]));
                nnet.assertFinite(d_err_w, "backprop out: d_err_w");
                train_result.d_w2[nidx] += d_err_w; // store result
            }
        }

        { // Hidden layer
            const h_len = @typeInfo(@TypeOf(self.h2)).Vector.len;
            // how much total error changes with respect to output (activated) of hidden layer
            //      𝝏err_total / 𝝏h
            var d_err_h: @Vector(h_len, Float) = undefined;
            for (self.w2, 0..) |w, nidx| {
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

            {
                // how much hidden layer (non activated) output change with respect to the weights:
                //      𝝏out / 𝝏w
                const d_o_w = self.h1;
                const d_tmp = d_err_h * d_h2_h2na;
                train_result.d_b2 += d_tmp; // store result
                for (self.w1, 0..) |_, nidx| {
                    // how much total error change with respect to the weights:
                    //      𝝏total_err / 𝝏w
                    const d_err_w = d_tmp * @as(@Vector(h_len, Float), @splat(d_o_w[nidx]));
                    nnet.assertFinite(d_err_w, "backprop hidden: d_err_w");
                    // store result
                    train_result.d_w1[nidx] += d_err_w;
                }
            }
        }
        train_result.correct += @as(u32, if (predicted_correct) 1 else 0);
        train_result.loss += total_err;
        train_result.test_cases += 1;
    }

    pub fn learn(self: *Self, train_results: TrainResult, learn_rate: Float) void {
        @setFloatMode(std.builtin.FloatMode.Optimized);
        self.bo += train_results.d_bo * @as(@Vector(@typeInfo(@TypeOf(self.bo)).Vector.len, Float), @splat(learn_rate));
        self.b2 += train_results.d_b2 * @as(@Vector(@typeInfo(@TypeOf(self.b2)).Vector.len, Float), @splat(learn_rate));
        for (self.w1, 0..) |w, nidx| {
            self.w1[nidx] -= train_results.d_w1[nidx] * @as(@Vector(@typeInfo(@TypeOf(w)).Vector.len, Float), @splat(learn_rate));
        }
        for (self.w2, 0..) |w, nidx| {
            self.w2[nidx] -= train_results.d_w2[nidx] * @as(@Vector(@typeInfo(@TypeOf(w)).Vector.len, Float), @splat(learn_rate));
        }
    }
};

pub fn doTest(alloc: mem.Allocator) !void {
    var net: NNet = undefined;
    if (options.load) |p| {
        var in_file = std.fs.cwd().openFile(p, .{}) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
        defer in_file.close();
        bin_file.readFile(NNet, &net, &in_file) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
    } else std.debug.panic("Can't test, network not specified! Use '--load' to specify network!", .{});

    var td = Dataset.init(alloc);
    const img_dir_path = "./data/digits/Images/test/";
    try td.load("./data/digits/test.csv", img_dir_path, "data/test.batch", false);
    mlog.info("Test data loaded: {} entries", .{td.test_cases.items.len});
    const in_dir = try fs.cwd().openDir(img_dir_path, .{});
    const out_dir: ?fs.Dir = odo: {
        if (options.img_dir_out) |p| {
            if (fs.cwd().openDir(p, .{})) |pdir| {
                pdir.makeDir("0") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("1") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("2") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("3") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("4") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("5") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("6") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("7") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("8") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                pdir.makeDir("9") catch |err| mlog.warn("Can't create directory for output! {}", .{err});
                break :odo pdir;
            } else |err| {
                mlog.err("Can't opon img_dir_out: '{s}' err: {}", .{ p, err });
            }
        }
        break :odo null;
    };

    { // iterate and test
        var of = try std.fs.cwd().createFile("res.csv", .{});
        defer of.close();
        const writer = of.writer();
        try writer.print("filename,label\n", .{});

        for (td.test_cases.items, 0..) |*test_case, i| {
            net.feedForward(&test_case.input);
            var best: u8 = 0;
            var best_confidence: Float = 0;
            var ti: usize = 0;
            while (ti < 10) : (ti += 1) {
                if (best_confidence < net.out_activated[ti]) {
                    best = @as(u8, @intCast(ti));
                    best_confidence = net.out_activated[ti];
                }
            }

            if (out_dir) |od| {
                var out_name_buf = [_]u8{0} ** 32;
                const out_name = std.fmt.bufPrint(out_name_buf[0..], "{}/{d:.0}%_{}#.png", .{ best, best_confidence * 100, i }) catch unreachable;
                in_dir.copyFile(td.getTestName(i), od, out_name, .{}) catch |err| mlog.err("Cant copy test output '{s}' err:{}", .{ out_name, err });
            }

            const test_name = td.test_names.items[i];
            try of.writer().print("{s},{}\n", .{ test_name, best });
            mlog.info("{s} , {} , {d:.1}%\t[{d:.2}]", .{ test_name, best, best_confidence * 100.0, net.out_activated * @as(@Vector(10, Float), @splat(100)) });
        }
    }
}

pub fn train(alloc: mem.Allocator) !void {
    var td = Dataset.init(alloc);
    defer td.deinit();
    try td.load("./data/digits/train.csv", "./data/digits/Images/train/", "data/train.batch", false);
    const seed = 364123;
    var rnd = std.rand.Sfc64.init(seed);
    var random_instance = rnd.random();
    const Trainer = @import("nnet_trainer.zig").forNet(NNet);
    var trainer = Trainer.init(alloc, random_instance);
    trainer.batch_size = options.batch_size;
    trainer.workers = options.workers;
    trainer.learn_rate = options.learn_rate;

    // load or initialise new net
    var net: NNet = undefined;
    if (options.load) |p| {
        var in_file = std.fs.cwd().openFile(p, .{}) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
        defer in_file.close();
        bin_file.readFile(NNet, &net, &in_file) catch |err| debug.panic("Can't open nnet: '{s}' Error:{}", .{ p, err });
    } else net.randomize(random_instance);

    // train
    var timer = try std.time.Timer.start();
    try trainer.trainEpoches(&net, &td.accessor, @as(u32, @intCast(options.epoches)));
    mlog.info("\nTotal train time: {}\n", .{fmtDuration(timer.lap())});

    // save net
    if (options.save) |p| {
        var in_file = std.fs.cwd().createFile(p, .{}) catch |err| debug.panic("Can't open file for storing nnet: '{s}' Error:{}", .{ p, err });
        defer in_file.close();
        bin_file.writeFile(NNet, &net, &in_file) catch |err| debug.panic("Can't write nnet to file: '{s}' Error:{}", .{ p, err });
    }
}

pub fn main() !void {
    LogCtx.init();
    defer LogCtx.deinit() catch debug.print("Can't flush!", .{});
    //LogCtx.testOut();

    options.workers = try std.Thread.getCpuCount();
    var galloc = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        switch (galloc.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }
    var lalloc = std.heap.LoggingAllocator(
        std.log.Level.debug,
        std.log.Level.info,
    ).init(galloc.allocator());
    var alloc = lalloc.allocator();

    { // ARGS
        const args = (try std.process.argsAlloc(galloc.allocator()))[1..]; // skip first arg as it points to current executable
        defer std.process.argsFree(galloc.allocator(), args);
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
        for (args, 0..) |argv, ai| {
            if (skip > 0) {
                skip -= 1;
                continue;
            }
            if (std.mem.eql(u8, argv, "--workers")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.workers = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--epoches")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.epoches = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--learn-rate")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by value!", .{argv});
                options.learn_rate = try std.fmt.parseFloat(Float, args[ai + 1]);
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--load")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.load = args[ai + 1];
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--save")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.save = args[ai + 1];
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--batch-size")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.batch_size = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--epoches")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.epoches = try std.fmt.parseUnsigned(usize, args[ai + 1], 0);
                skip = 1;
            } else if (std.mem.eql(u8, argv, "--img-dir-out")) {
                if (ai + 1 >= args.len) std.debug.panic("Argument '{s}' needs to be followed by path!", .{argv});
                options.img_dir_out = args[ai + 1];
                skip = 1;
            } else if (std.mem.eql(u8, argv, "preprocess")) {
                {
                    mlog.info("Preprocessing training data...", .{});
                    var td = Dataset.init(alloc);
                    defer td.deinit();
                    td.load("./data/digits/train.csv", "./data/digits/Images/train/", "data/train.batch", true) catch |err| debug.panic("Error: {}", .{err});
                    td.saveBatch("data/train.batch") catch |err| debug.panic("Error: {}", .{err});
                }
                {
                    mlog.info("Preprocessing test data...", .{});
                    var td = Dataset.init(alloc);
                    defer td.deinit();
                    td.load("./data/digits/test.csv", "./data/digits/Images/test/", "data/test.batch", true) catch |err| debug.panic("Error: {}", .{err});
                    td.saveBatch("data/test.batch") catch |err| debug.panic("Error: {}", .{err});
                }
            } else if (std.mem.eql(u8, argv, "train")) {
                train(alloc) catch |err| debug.panic("Error: {}", .{err});
            } else if (std.mem.eql(u8, argv, "test")) {
                try doTest(alloc);
            } else if (std.mem.eql(u8, argv, "help")) {
                printHelp();
            } else std.debug.panic("Unknown argument: {s}", .{argv});
        }
    }
}
