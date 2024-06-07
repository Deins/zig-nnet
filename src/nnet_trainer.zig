const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const atomic = std.atomic;
const maxInt = std.math.maxInt;
const SpinFutex = struct {
    // spin-lock
    pub fn wait(ptr: *const atomic.Value(u32), expect: u32, comptime timeout: ?u64) error{TimedOut}!void {
        if (timeout) |_| @compileError("Not implemented!");
        while (ptr.load(.Unordered) == expect) {
            std.os.sched_yield() catch std.atomic.spinLoopHint();
        }
    }
    pub fn wake(ptr: *const atomic.Value(u32), num_waiters: u32) void {
        _ = ptr;
        _ = num_waiters;
    }
};
//const Futex = SpinFutex;
const Futex = std.Thread.Futex;
const nnet = @import("nnet.zig");

// multithreaded nnet trainer
pub fn forNet(comptime NNet: anytype) type {
    return struct {
        const Self = @This();
        pub const Float = NNet.ValType;
        pub const TestCase = nnet.typed(NNet.ValType).TestCase(NNet.input_len, NNet.output_len);
        pub const TestAccessor = nnet.typed(NNet.ValType).TestAccessor(TestCase);
        const log = std.log.scoped(.NNetTrainer);
        // cordinates workers and work to be done
        const WorkQueue = struct {
            const stop = std.math.maxInt(u32);
            const start = stop - 1;
            inputs: []u32 = undefined,
            next_input: atomic.Value(u32) = atomic.Value(u32).init(start),
            results_done: atomic.Value(u32) = atomic.Value(u32).init(0),
            results_mutex: std.Thread.Mutex = .{},
            results: NNet.TrainResult = .{},
        };
        alloc: mem.Allocator,
        rnd: std.rand.Random,
        batch_size: usize = 4,
        learn_rate: Float = 0.1,
        epoch_idx: usize = 0,
        workers: usize = 8,
        // private - used to coordinate threads

        pub fn init(alloc: mem.Allocator, rnd: std.rand.Random) Self {
            return .{
                .alloc = alloc,
                .rnd = rnd,
            };
        }

        export fn worker(wq: *WorkQueue, inp_nnet: *NNet, tacc: *TestAccessor) void {
            @setFloatMode(.optimized);
            var net = inp_nnet.*;
            var r: NNet.TrainResult = .{};
            Futex.wait(&wq.next_input, WorkQueue.start);
            var inputs: []u32 = wq.inputs;
            main_worker_loop: while (true) {
                const input_idx = wq.next_input.fetchAdd(1, .acq_rel);
                if (input_idx < inputs.len) {
                    //var timer = std.time.Timer.start() catch unreachable;
                    const test_case = tacc.grabTest(inputs[input_idx]);
                    defer tacc.freeTest(test_case);
                    net.trainDeriv(test_case.*, &r);
                    //std.debug.print("trainDeriv: {}\n", .{std.fmt.fmtDuration(timer.lap())});
                } else {
                    // no more inputs
                    if (r.test_cases > 0) {
                        // merge whats left of results
                        wq.results_mutex.lock();
                        wq.results.merge(r);
                        const total = wq.results.test_cases;
                        wq.results_mutex.unlock();
                        const total_i: u32 = @as(u32, @intFromFloat(total));
                        wq.results_done.store(total_i, .release);
                        if (total_i == @as(u32, @intCast(inputs.len))) {
                            Futex.wake(&wq.results_done, maxInt(u32));
                            //log.info("wake: {}", .{total_i});
                        } // else log.info("NO wake: {}", .{total_i});
                        r = .{}; // fresh working copy
                    }
                    { // go to sleep
                        var inp: u32 = wq.next_input.load(.acquire);
                        while (inp >= inputs.len) {
                            if (inp == WorkQueue.stop) {
                                break :main_worker_loop; // all done - exit
                            }
                            Futex.wait(&wq.next_input, inp);
                            if (input_idx > wq.next_input.load(.acquire)) break;
                            inp = wq.next_input.load(.acquire);
                        }
                    }
                    // prep for work
                    net = inp_nnet.*; // copy fresh working copy
                    inputs = wq.inputs;
                }
            }
        }

        pub fn trainEpoches(self: *Self, net: *NNet, test_accesor: *TestAccessor, epoches: u32) !void {
            @setFloatMode(.optimized);
            const workers: usize = @min(self.workers, self.batch_size);
            if (workers < 1 or workers > 1024) std.debug.panic("Invalid worker count: {}", .{workers});
            std.debug.print("Epoch {} started (batchsize: {}, threads: {})\n", .{ self.epoch_idx, self.batch_size, self.workers });
            var timer = try std.time.Timer.start();
            const test_len: usize = test_accesor.testCount();
            // TODO: move as struct member to avoid allocating for each epoch
            var shuffled: []u32 = try self.alloc.alloc(u32, test_len);
            defer self.alloc.free(shuffled);
            for (shuffled, 0..) |*e, idx| {
                e.* = @as(u32, @intCast(idx));
            }

            var wq = try self.alloc.create(WorkQueue);
            defer self.alloc.destroy(wq);
            wq.* = .{};
            const threads = try self.alloc.alloc(std.Thread, workers);
            defer self.alloc.free(threads);

            for (threads) |*t| t.* = try std.Thread.spawn(.{ .stack_size = 1671168 + @sizeOf(NNet) }, worker, .{ wq, net, test_accesor });

            var print_timer = try std.time.Timer.start();
            var epoch_i: u32 = 0;
            while (epoch_i < epoches) : (epoch_i += 1) {
                defer self.epoch_idx += 1;
                { // shuffle
                    var st = try std.time.Timer.start();
                    self.rnd.shuffle(@TypeOf(shuffled[0]), shuffled);
                    log.info("Shuffle time: {}", .{std.fmt.fmtDuration(st.read())});
                }

                wq.* = .{};
                var correct: u32 = 0.0;
                var loss: Float = 0.0;
                var bb: usize = 0;
                while (bb < test_len) : (bb += self.batch_size) {
                    const batch = shuffled[bb..@min(bb + self.batch_size, test_len)];
                    if (wq.results_mutex.tryLock()) {
                        defer wq.results_mutex.unlock();
                        wq.results = .{};
                    } else @panic("Threading error: someone has grabbed results lock!");
                    wq.results_done.store(0, .release);
                    wq.inputs = batch;
                    wq.next_input.store(@as(u32, @intCast(0)), .release);
                    Futex.wake(&wq.next_input, maxInt(u32)); // wake workers
                    // wait for result
                    var expect_res: u32 = 0;
                    while (expect_res != batch.len) {
                        Futex.wait(&wq.results_done, expect_res);
                        expect_res = wq.results_done.load(.acquire);
                        //log.info("Results: {}", .{expect_res});
                    }
                    // update nnet
                    // note: we don't lock wq.results as at this point workers are sleeping
                    if (@as(usize, @intFromFloat(wq.results.test_cases)) != batch.len)
                        std.debug.panic("Bad threading - result count doesn't match expected batch size! {}/{}", .{ @as(usize, @intFromFloat(wq.results.test_cases)), batch.len });
                    //wq.results.average();
                    net.learn(wq.results, self.learn_rate);
                    if (!std.math.isFinite(wq.results.loss)) {
                        log.err("Batch {}# loss not finite! {}", .{ bb, wq.results.loss });
                        //wq.results.print();
                        @panic("loss not finite!");
                    }
                    loss += wq.results.loss;
                    correct += wq.results.correct;
                    if (print_timer.read() > 50 * 1000 * 1000) {
                        print_timer.reset();
                        const acc = 100 * @as(Float, @floatFromInt(wq.results.correct)) / @as(Float, @floatFromInt(batch.len));
                        log.info("Batch {}# loss: {d:.4} accuracy: {d:.2}%", .{ bb / self.batch_size, wq.results.loss, acc });
                    }
                }
                // print stats
                const accuracy = @as(Float, @floatFromInt(correct)) / @as(Float, @floatFromInt(test_len));
                log.info("Epoch {}# train time: {} avg loss: {d:.4} accuracy: {d:.2}%\n", .{ self.epoch_idx, std.fmt.fmtDuration(timer.lap()), loss / @as(Float, @floatFromInt(test_len)), accuracy * 100 });
            }
            // stop workers
            wq.next_input.store(WorkQueue.stop, .seq_cst);
            Futex.wake(&wq.next_input, maxInt(u32));
            for (threads) |t| t.join();
        }
    };
}
