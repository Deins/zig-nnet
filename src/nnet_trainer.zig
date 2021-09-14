const std = @import("std");
const mem = std.mem;
const atomic = std.atomic;
const maxInt = std.math.maxInt;
const SpinFutex = struct {
    // spin-lock
    pub fn wait(ptr: *const atomic.Atomic(u32), expect: u32, comptime timeout: ?u64) error{TimedOut}!void {
        if (timeout) |_| @compileError("Not implemented!");
        while (ptr.load(.Unordered) == expect) {
            std.os.sched_yield() catch std.atomic.spinLoopHint();
        }
    }
    pub fn wake(ptr: *const atomic.Atomic(u32), num_waiters: u32) void {
        _ = ptr;
        _ = num_waiters;
    }
};
//const Futex = SpinFutex;
const Futex = std.Thread.Futex;
const nnet = @import("nnet.zig");

// multithreaded nnet trainer
pub fn forNet(NNet: anytype) type {
    return struct {
        const Self = @This();
        pub const Float = NNet.ValType;
        pub const TestCase = nnet.typed(NNet.ValType).TestCase(NNet.input_len, NNet.output_len);
        pub const TestAccessor = nnet.typed(NNet.ValType).TestAccessor(TestCase);
        const log = std.log.scoped(.NNetTrainer);
        // cordinates workers and work to be done
        const WorkQueue = struct {
            const stop = maxInt(u32);
            const start = stop - 1;
            inputs: []u32 = undefined,
            next_input: atomic.Atomic(u32) = .{ .value = start },
            results_done: atomic.Atomic(u32) = .{ .value = 0 },
            results_mutex: std.Thread.Mutex = .{},
            results: NNet.TrainResult = .{},
        };
        alloc: *mem.Allocator,
        rnd: *std.rand.Random,
        batch_size: usize = 4,
        learn_rate: Float = 0.1,
        epoch_idx: usize = 0,
        workers: usize = 8,
        // private - used to coordinate threads

        pub fn init(alloc: *mem.Allocator, rnd: *std.rand.Random) Self {
            return .{
                .alloc = alloc,
                .rnd = rnd,
            };
        }

        fn worker(wq: *WorkQueue, inp_nnet: *NNet, tacc: *TestAccessor) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            var net = inp_nnet;
            var r: NNet.TrainResult = .{};
            Futex.wait(&wq.next_input, WorkQueue.start, null) catch unreachable;
            var inputs : []u32 = wq.inputs;
            main_worker_loop: while (true) {
                var input_idx = wq.next_input.fetchAdd(1, atomic.Ordering.AcqRel);
                if (input_idx < inputs.len) {
                    //var timer = std.time.Timer.start() catch unreachable;
                    var test_case = tacc.grabTest(inputs[input_idx]);
                    defer tacc.freeTest(test_case);
                    net.trainDeriv(test_case.*, &r);
                    //std.debug.print("trainDeriv: {}\n", .{std.fmt.fmtDuration(timer.lap())});
                } else {
                    // no more inputs
                    if (r.test_cases > 0) {
                        // merge whats left of results
                        var lock = wq.results_mutex.acquire();
                        wq.results.merge(r);
                        var total = wq.results.test_cases;
                        lock.release();
                        var total_i: u32 = @floatToInt(u32, total);
                        wq.results_done.store(total_i, .Release);
                        if (total_i == @intCast(u32, inputs.len)) {
                            Futex.wake(&wq.results_done, maxInt(u32));
                            //log.info("wake: {}", .{total_i});
                        } // else log.info("NO wake: {}", .{total_i});
                        r = .{}; // fresh working copy
                    }
                    { // go to sleep
                        var inp: u32 = wq.next_input.load(.Acquire);
                        while (inp >= inputs.len) {
                            if (inp == WorkQueue.stop) {
                                break :main_worker_loop; // all done - exit
                            }
                            Futex.wait(&wq.next_input, inp, null) catch unreachable;
                            if (input_idx > wq.next_input.load(atomic.Ordering.Acquire)) break;
                            inp = wq.next_input.load(.Acquire);
                        }
                    }
                    // prep for work
                    net = inp_nnet; // copy fresh working copy
                    inputs = wq.inputs;
                }
            }
        }

        pub fn trainEpoches(self: *Self, net: *NNet, test_accesor: *TestAccessor, epoches: u32) !void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            const workers: usize = @minimum(self.workers, self.batch_size);
            if (workers < 1 or workers > 1024) std.debug.panic("Invalid worker count: {}", .{workers});
            std.debug.print("Epoch {} started (batchsize: {}, threads: {})\n", .{ self.epoch_idx, self.batch_size, self.workers });
            var timer = try std.time.Timer.start();
            const test_len: usize = test_accesor.testCount();
            // TODO: move as struct member to avoid allocating for each epoch
            var shuffled: []u32 = try self.alloc.alloc(u32, test_len);
            defer self.alloc.free(shuffled);
            for (shuffled) |*e, idx| { e.* = @intCast(u32,idx); }

            var wq = try self.alloc.create(WorkQueue);
            defer self.alloc.destroy(wq);
            wq.* = .{};
            var threads = try self.alloc.alloc(std.Thread, workers);
            defer self.alloc.free(threads);

            for (threads) |*t| t.* = try std.Thread.spawn(.{ .stack_size = 1671168 + @sizeOf(NNet) }, worker, .{ wq, net, test_accesor});

            var print_timer = try std.time.Timer.start();
            var epoch_i: u32 = 0;
            while (epoch_i < epoches) : (epoch_i += 1) {
                defer self.epoch_idx += 1;
                {   // shuffle
                    var st = try std.time.Timer.start();
                    self.rnd.shuffle(@TypeOf(shuffled[0]), shuffled);
                    log.info("Shuffle time: {}", .{std.fmt.fmtDuration(st.read())});
                }

                wq.* = .{};
                var correct: u32 = 0.0;
                var loss: Float = 0.0;
                var bb: usize = 0;
                while (bb < test_len) : (bb += self.batch_size) {
                    const batch = shuffled[bb..@minimum(bb + self.batch_size, test_len)];
                    var lock = wq.results_mutex.tryAcquire();
                    if (lock) |l| {
                        wq.results = .{};
                        l.release();
                    } else @panic("Threading error: someone has grabbed results lock!");
                    wq.results_done.store(0, .Release);
                    wq.inputs = batch;
                    wq.next_input.store(@intCast(u32, 0), .Release);
                    Futex.wake(&wq.next_input, maxInt(u32)); // wake workers
                    // wait for result
                    var expect_res: u32 = 0;
                    while (expect_res != batch.len) {
                        Futex.wait(&wq.results_done, expect_res, null) catch unreachable;
                        expect_res = wq.results_done.load(.Acquire);
                        //log.info("Results: {}", .{expect_res});
                    }
                    // update nnet
                    // note: we don't lock wq.results as at this point workers are sleeping
                    if (@floatToInt(usize, wq.results.test_cases) != batch.len)
                        std.debug.panic("Bad threading - result count doesn't match expected batch size! {}/{}", .{ @floatToInt(usize, wq.results.test_cases), batch.len });
                    //wq.results.average();
                    net.learn(wq.results, self.learn_rate);
                    if (!std.math.isFinite(wq.results.loss)) {
                        log.alert("Batch {}# loss not finite! {}", .{bb, wq.results.loss});
                        //wq.results.print();
                        std.os.abort();
                    }
                    loss += wq.results.loss;
                    correct += wq.results.correct;
                    if (print_timer.read() > 50 * 1000 * 1000) 
                    {
                        print_timer.reset();
                        const acc = 100 * @intToFloat(Float, wq.results.correct) / @intToFloat(Float, batch.len);
                        log.info("Batch {}# loss: {d:.4} accuracy: {d:.2}%", .{ bb / self.batch_size, wq.results.loss, acc });
                    }
                }
                // print stats
                const accuracy = @intToFloat(Float, correct) / @intToFloat(Float, test_len);
                log.notice("Epoch {}# train time: {} avg loss: {d:.4} accuracy: {d:.2}%\n", .{ self.epoch_idx, std.fmt.fmtDuration(timer.lap()), loss / @intToFloat(Float, test_len), accuracy * 100 });
            }
            // stop workers
            wq.next_input.store(WorkQueue.stop, .SeqCst);
            Futex.wake(&wq.next_input, maxInt(u32));
            for (threads) |t| t.join();
        }
    };
}
