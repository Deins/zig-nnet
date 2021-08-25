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
const Futex = std.Thread.Futex;
//const Futex = SpinFutex;

// multithreaded nnet trainer
pub fn forNet(NNet: anytype) type {
    const TestAccessor = @import("tdata.zig").forData(NNet.ValType, NNet.input_len, NNet.output_len).TestAccessor;
    return struct {
        const Self = @This();
        const Float = NNet.ValType;
        const log = std.log.scoped(.NNetTrainer);
        // cordinates workers and work to be done
        // flow:
        //  1. worker threads are spin up and they initialise (make nnet working copy) and wait on current_batch_futex (with expect = maxInt(u32))
        //  2. main thread initialises batch and wake current_batch_futex
        //  3. main thread waits results_done futex
        //  4. main thread merges each result (or waits on futex)
        //  5. when worker wakes up, each worker grabs input by increasing next_input and grabbing coresponding item from inputs
        //  6. worker processes inputs and updates results by placing it in coresponding index and increasing results_done then wakes results_done
        //  7. worker then repeats with grab new input, if all are consumed, sleeps on batch_idx to start from step 5
        //  8. main thread when all results are done updates nnet (each worker makes copy when new batch starts)
        //  9. main thread repeats with new batch (step 2), or if no work is left wakes futex current_batch_futex maxInt(u32) and joins threads
        const WorkQueue = struct {
            active_workers: atomic.Atomic(u32) = .{ .value = 0 },
            current_batch_futex: atomic.Atomic(u32) = .{ .value = maxInt(u32) }, // stores batch id
            next_input: atomic.Atomic(u32) = .{ .value = 0 },
            inputs: []usize = undefined,
            results_mutex: std.Thread.Mutex = .{},
            results: NNet.TrainResult = undefined,
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

        fn worker(wq: *WorkQueue, inp_nnet: *NNet, td: *TestAccessor) void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            var nnet = inp_nnet;
            Futex.wait(&wq.current_batch_futex, maxInt(u32), null) catch |err| {
                std.debug.panic("Worker wait failed with err: {}", .{err});
            };
            _ = wq.active_workers.fetchAdd(1, atomic.Ordering.Monotonic);
            Futex.wake(&wq.active_workers, maxInt(u32));
            var current_batch = wq.current_batch_futex.load(.Monotonic); // note: should be safe even thread skips a batch in extreme case as wait doesn't return new batch value
            while (true) {
                const input_idx = wq.next_input.fetchAdd(1, atomic.Ordering.Monotonic);
                if (input_idx < wq.inputs.len) {
                    //var timer = std.time.Timer.start() catch unreachable;
                    var r: NNet.TrainResult = undefined;
                    nnet.trainDeriv(td.getTest(wq.inputs[input_idx]), &r);
                    var lock = wq.results_mutex.acquire();
                    wq.results.merge(r);
                    lock.release();
                    //std.debug.print("trainDeriv: {}\n", .{std.fmt.fmtDuration(timer.lap())});
                } else {
                    // no more inputs - wait for different batch
                    _ = wq.active_workers.fetchSub(1, atomic.Ordering.Monotonic);
                    Futex.wake(&wq.active_workers, maxInt(u32));
                    Futex.wait(&wq.current_batch_futex, current_batch, null) catch |err| {
                        std.debug.panic("Worker wait failed with err: {}", .{err});
                    };
                    current_batch = wq.current_batch_futex.load(.Monotonic);
                    if (current_batch == maxInt(u32)) // all jobs done - exit
                        break;
                    // still more work to do
                    _ = wq.active_workers.fetchAdd(1, atomic.Ordering.Monotonic);
                    Futex.wake(&wq.active_workers, maxInt(u32));
                    nnet = inp_nnet; // copy fresh working copy
                }
            }
        }

        pub fn trainEpoch(self: *Self, nnet: *NNet, td: *TestAccessor) !void {
            @setFloatMode(std.builtin.FloatMode.Optimized);
            defer self.epoch_idx += 1;
            const workers: usize = @minimum(self.workers, self.batch_size);
            std.debug.print("Epoch {} started (batchsize: {}, threads: {})\n", .{ self.epoch_idx, self.batch_size, self.workers });
            var timer = try std.time.Timer.start();
            const test_len: usize = td.getLen();
            // TODO: move as struct member to avoid allocating for each epoch
            var shuffled: []usize = try self.alloc.alloc(usize, test_len);
            defer self.alloc.free(shuffled);

            // shuffle
            for (shuffled) |*e, idx| e.* = idx;
            self.rnd.shuffle(usize, shuffled);

            var wq = try self.alloc.create(WorkQueue);
            defer self.alloc.destroy(wq);
            wq.* = .{};
            var threads = try self.alloc.alloc(std.Thread, workers);
            defer self.alloc.free(threads);

            for (threads) |*t| t.* = try std.Thread.spawn(.{ .stack_size = 1671168 + @sizeOf(NNet) }, worker, .{ wq, nnet, td });

            var correct: u32 = 0.0;
            var loss: Float = 0.0;
            var bb: usize = 0;
            while (bb < test_len) : (bb += self.batch_size) {
                const batch = shuffled[bb..@minimum(bb + self.batch_size, test_len)];
                wq.results = .{};
                wq.inputs = batch;
                wq.next_input.store(0, .Monotonic);
                wq.current_batch_futex.store(@intCast(u32, bb), .Monotonic);
                Futex.wake(&wq.current_batch_futex, maxInt(u32)); // wake workers
                // wait for threads to spin up
                Futex.wait(&wq.active_workers, 0, null) catch |err| {
                    std.debug.panic("Futex can't wait! err: {}", .{err});
                };
                { // wait for threads to finish
                    var expect: u32 = wq.active_workers.load(.Acquire);
                    while (expect != 0) {
                        Futex.wait(&wq.active_workers, expect, null) catch |err| {
                            std.debug.panic("Futex can't wait! err: {}", .{err});
                        };
                        expect = wq.active_workers.load(.Acquire);
                    }
                }
                // update nnet
                // note: we don't lock wq.results as at this point workers are sleeping
                wq.results.finalizeMerge();
                nnet.learn(wq.results, self.learn_rate);
                loss += wq.results.loss;
                correct += wq.results.correct;
                const acc = 100 * @intToFloat(Float, wq.results.correct) / @intToFloat(Float, batch.len);
                log.info("Batch {}# loss: {d:.4} accuracy: {d:.2}%", .{ bb / self.batch_size, wq.results.loss, acc });
            }
            // stop workers
            wq.current_batch_futex.store(maxInt(u32), .Monotonic);
            Futex.wake(&wq.current_batch_futex, maxInt(u32));
            for (threads) |t| t.join();

            // var res = try self.alloc.alloc(NNet.TrainResult, self.threads + 2);
            // defer self.alloc.free(res);
            // var res_tmp = &res[res.len - 2];
            // var res_total = &res[res.len - 1];
            // var tid: usize = 0;
            // var correct: u32 = 0.0;
            // var frames = try self.alloc.alloc(@Frame(NNet.trainDeriv), self.threads);
            // defer self.alloc.free(frames);
            // while (tid < test_len) : (tid += self.batch_size) {
            //     const batch = shuffled[tid..@minimum(tid + self.batch_size, test_len)];
            //     std.debug.print("Batch: {} {}\n", .{ tid, tid + batch.len });
            //     res_tmp.* = .{};
            //     var thread : usize = 0;
            //     for (batch) |bt, bi| {
            //         std.Thread.spawn(.{}, nnet.trainDeriv, .{nnet, td.getTest(bt), &res[thread]});
            //         frames[thread] = async nnet.trainDeriv();
            //         thread += 1;
            //         if (thread >= self.threads) {
            //             // merge
            //             while (thread > 0) : (thread -= 1) {
            //                 await frames[thread - 1];
            //                 if (res[thread -1].correct) correct += 1;
            //                 res_tmp.merge(res[thread - 1]);
            //             }
            //             thread = 0;
            //         }
            //         if (bi > 0) res[1].merge(res[0]) else res[1] = res[0];
            //     }
            //     // merge
            //     while (thread > 0) : (thread -= 1) {
            //         await frames[thread - 1];
            //         if (res[thread -1].correct) correct += 1;
            //         res_tmp.merge(res[thread - 1]);
            //     }
            //     thread = 0;
            //     thread = 0;

            //     res_total.merge(res[1]);
            //     res_tmp.finalizeMerge();
            //     nnet.learn(res[1], 0.1);
            // }

            const accuracy = @intToFloat(Float, correct) / @intToFloat(Float, test_len);
            std.debug.print("\nEpoch {}# train time: {} avg loss: {d:.4} accuracy: {d:.2}%\n", .{ self.epoch_idx, std.fmt.fmtDuration(timer.lap()), loss / @intToFloat(Float, test_len), accuracy * 100 });
        }
    };
}
