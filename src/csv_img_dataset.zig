const std = @import("std");
const mem = std.mem;
const os = std.os;
const io = std.io;
const fs = std.fs;
const csv = @import("csv");
//const fmapper = @import("fmapper.zig");
const log_ctx = @import("log.zig");
const fmtDuration = std.fmt.fmtDuration;
const ansi = @import("ansi_esc.zig");
const heap = std.heap;

const c = @cImport({
    @cInclude("stb_image.h");
});

pub fn forData(comptime Float: type, comptime input_size: [2]usize, comptime output_len: usize) type {
    const nnet = @import("nnet.zig").typed(Float);
    return struct {
        const Self = @This();
        pub const file_header: [8]u8 = [_]u8{ 'I', 'M', 'G', 'T', 'D', 'A', 'T', 'A' };
        pub const file_version = 0;
        pub const TestCase = nnet.TestCase(input_size[0] * input_size[1], output_len);
        pub const TestAccessor = nnet.TestAccessor(TestCase);
        pub const log = std.log.scoped(.csv_img_dataset);
        pub const max_name_len = 31;

        accessor: TestAccessor,
        arena: heap.ArenaAllocator,
        // arrays skip arena to be able to actually
        test_cases: std.ArrayList(TestCase),
        test_names: std.ArrayList([max_name_len:0]u8),

        pub fn init(alloc: *std.mem.Allocator) Self {
            return .{
                .arena = heap.ArenaAllocator.init(alloc),
                .test_cases = std.ArrayList(TestCase).init(alloc),
                .test_names = std.ArrayList([max_name_len:0]u8).init(alloc),
                .accessor = .{ .grabFn = Self.getTest, .countFn = getLen },
            };
        }

        pub fn deinit(self: *Self) void {
            self.test_cases.deinit();
            self.test_names.deinit();
            self.arena.deinit();
        }

        pub fn load(self: *Self, csv_path: []const u8, img_dir_path: []const u8, batch_path: []const u8, from_source_only: bool) !void {
            if (!from_source_only) {
                if (self.loadBatch(batch_path)) {
                    return; // success
                } else |err| log.warn("Could not load images from batch: {} : fallback to load from sources", .{err});
            }

            // load from sources
            try self.loadCsv(csv_path);
            try self.loadImages(img_dir_path);
        }

        pub fn loadCsv(self: *Self, path: []const u8) !void {
            log.info("Loading csv '{s}'", .{path});
            var timer = try std.time.Timer.start();
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            const buffer = try self.arena.child_allocator.alloc(u8, try file.getEndPos());
            defer self.arena.child_allocator.free(buffer);

            const csv_tokenizer = &try csv.CsvTokenizer(std.fs.File.Reader).init(file.reader(), buffer, .{});
            const aprox_capacity = buffer.len / 10;
            try self.test_cases.ensureTotalCapacity(aprox_capacity);
            try self.test_names.ensureTotalCapacity(aprox_capacity);

            const err = error{ CsvExpectsOnlyTwoColumns, CsvMissingColumn };

            const cols = read_col: { // read header
                var ci: usize = 0;
                while (try csv_tokenizer.next()) |tok| : (ci += 1) {
                    if (tok == .row_end) break;
                }
                break :read_col ci;
            };
            if (cols < 1) {
                log.err("Dataset csv requires at least 1 column. cols in csv: {}", .{cols});
                return error.WrongHeaderCSV;
            }
            if (cols > 2) {
                log.warn("Dataset uses only 1 or 2 columns! cols: {}", .{cols});
            }

            var i: i32 = 0;
            while (try csv_tokenizer.next()) |first_token| : (i += 1) {
                const tc = try self.test_cases.addOne();
                const tn = try self.test_names.addOne();
                std.mem.copy(u8, tn, first_token.field[0..@min(first_token.field.len, max_name_len)]);
                tn[first_token.field.len] = 0;
                if (cols > 1) {
                    if (try csv_tokenizer.next()) |tok| {
                        if (tok != .field) {
                            log.alert("Expected field in csv, got end of line!", .{});
                            return error.MissingValueCSV;
                        }
                        const digit: u8 = std.fmt.parseInt(u8, tok.field, 10) catch |e| {
                            log.alert("CSV can't parse nubmer: `{s}` err: {}", .{ first_token.field, e });
                            return error.ExpectedNumberCSV;
                        };
                        var answer_vec: []Float = @splat(0);
                        answer_vec[digit] = 1.0;
                        tc.*.answer = answer_vec;
                    } else return err.CsvMissingColumn;
                }
                // skip remaining columns
                while (try csv_tokenizer.next()) |tok| {
                    if (tok == .row_end) break;
                }
            }
            log.info("CSV loaded {} records in {}", .{ self.test_cases.items.len, fmtDuration(timer.lap()) });
        }

        pub fn loadImages(self: *Self, path_prefix: []const u8) !void {
            var timer = try std.time.Timer.start();
            log.info("Loading images from sources...", .{});
            const prcent_mod = self.test_cases.items.len / 100;
            var path_buff: [512]u8 = undefined;
            mem.copy(u8, path_buff[0..], path_prefix);

            var tmp_alloc_buff = try self.arena.child_allocator.alloc(u8, input_size[0] * input_size[1] * 4 + 1024);
            defer self.arena.child_allocator.free(tmp_alloc_buff);
            var tmp_alloc = heap.FixedBufferAllocator.init(tmp_alloc_buff);

            for (self.test_cases.items, 0..) |_, i| {
                if (i % prcent_mod == 0) {
                    log.info("Image loading progress: {}%\r", .{i * 100 / self.test_cases.items.len});
                    log_ctx.log_ctx.out.flush() catch {};
                }

                const img_name = self.getTestName(i);
                mem.copy(u8, path_buff[path_prefix.len..], img_name[0..]);
                const path = path_buff[0..(path_prefix.len + img_name.len)];
                //debug.print("Loading image `{s}`\n", .{path});

                // read whole file
                const file = std.fs.cwd().openFile(path, .{}) catch |err| {
                    std.debug.panic("Error opening image '{s}': {}", .{ path, err });
                };
                defer file.close();
                const buffer = try tmp_alloc.allocator.alloc(u8, try file.getEndPos());
                defer tmp_alloc.allocator.free(buffer);
                const read_len = try file.readAll(buffer);
                if (read_len != buffer.len) std.debug.panic("Read different amount of bytes than in file!", .{});

                //const map = try fmapper.FMapper.open(path, .{ .mode = .shared }, 0, 0);
                //defer map.close();
                //const buffer = map.memory.?;

                var w: i32 = 0;
                var h: i32 = 0;
                var chanells_in_file: i32 = 1;
                const desired_channels: i32 = 1;
                const pixels = c.stbi_load_from_memory(buffer.ptr, @as(i32, @intCast(buffer.len)), &w, &h, &chanells_in_file, desired_channels);
                defer std.c.free(pixels);
                if (pixels == null) {
                    std.debug.panic("Can't load image {s} filesize: {}", .{ path, buffer.len });
                }
                if (desired_channels == chanells_in_file) {
                    std.debug.panic("Loaded image has different pixel format: conversion not implemented! {}/{}", .{ desired_channels, chanells_in_file });
                }
                if (w != input_size[0] or h != input_size[1]) {
                    std.debug.panic("Wrong image `{s}` size: expected {}x{} got {}x{}", .{ img_name, input_size[0], input_size[1], w, h });
                }

                for (pixels[0..(input_size[0] * input_size[1])], 0..) |pixel, i_pix| {
                    self.test_cases.items[i].input[i_pix] = @as(Float, @floatFromInt(pixel)) * (1.0 / 255.0);
                }
            }
            log.info("Images loaded from sources in {}", .{fmtDuration(timer.lap())});
        }

        pub fn getTestName(self: *Self, idx: usize) []const u8 {
            return mem.sliceTo(&self.test_names.items[idx], 0);
        }

        // saves batch training data in single file for faster loading
        pub fn saveBatch(self: *Self, path: []const u8) !void {
            log.info("Saving '{s}' batch...", .{path});
            const f = try std.fs.cwd().createFile(path, .{ .truncate = true });
            defer f.close();
            var w = f.writer();
            try w.writeAll(file_header[0..]);
            try w.writeIntLittle(u32, file_version);
            try w.writeIntLittle(u32, input_size[0]);
            try w.writeIntLittle(u32, input_size[1]);

            try w.writeIntLittle(u64, self.test_cases.items.len);
            if (comptime std.Target.current.cpu.arch.endian() != .Little) {
                @panic("TODO: Implement endian conversion!");
            }
            try w.writeAll(std.mem.sliceAsBytes(self.test_names.items));
            try w.writeAll(std.mem.sliceAsBytes(self.test_cases.items));
            log.info("Batch '{s}' saved for future speedup. Run preprocess when source data is changed!", .{path});
        }

        // loads batched training data
        pub fn loadBatch(self: *Self, path: []const u8) !void {
            log.info("Loading batch '{s}'...", .{path});
            var timer = try std.time.Timer.start();
            const f = try std.fs.cwd().openFile(path, .{});
            var r = f.reader();
            var header: [file_header.len]u8 = undefined;
            try r.readNoEof(&header);
            const err = error{ HeaderMismatch, VersionMismatch, ImageSizeMismatch };
            if (!mem.eql(u8, &header, &file_header)) {
                return err.HeaderMismatch;
            }
            const v = try r.readIntLittle(u32);
            const w = try r.readIntLittle(u32);
            const h = try r.readIntLittle(u32);

            if (v != file_version) return err.VersionMismatch;
            if (w != input_size[0] or h != input_size[1]) return err.ImageSizeMismatch;

            var records: u64 = try r.readIntLittle(u64);
            if (comptime std.Target.current.cpu.arch.endian() != .Little) {
                @compileError("TODO: Implement endian conversion!");
            }
            try self.test_names.resize(records);
            try r.readNoEof(std.mem.sliceAsBytes(self.test_names.items));
            try self.test_cases.resize(records);
            try r.readNoEof(std.mem.sliceAsBytes(self.test_cases.items));
            log.info("Batch loaded {} records in {}", .{ records, fmtDuration(timer.read()) });
        }

        //  TestAccessor funcs - private
        fn getTest(a: *TestAccessor, idx: usize) *TestCase {
            const self = @fieldParentPtr(Self, "accessor", a);
            return &self.test_cases.items[idx];
        }

        fn getLen(a: *TestAccessor) usize {
            const self = @fieldParentPtr(Self, "accessor", a);
            return self.test_cases.items.len;
        }
    };
}
