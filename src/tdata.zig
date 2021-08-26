const std = @import("std");
const mem = std.mem;
const os = std.os;
const debug = std.debug;
const io = std.io;
const json = std.json;
const fs = std.fs;
const csv = @import("csv");
const fmapper = @import("fmapper.zig");
const fmtDuration = std.fmt.fmtDuration;

const c = @cImport({
    @cInclude("stb_image.h");
});

pub fn forData(comptime Float: type, comptime input_len: usize, comptime output_len: usize) type {
    return struct {
        pub const TestCase = struct { name: []u8, input: @Vector(input_len, Float), answer: @Vector(output_len, Float) };

        pub const TestAccessor = struct {
            const Self = @This();
            lenFn: fn (s: *Self) usize,
            testFn: fn (s: *Self, idx: usize) TestCase,
            pub fn getLen(self: *Self) usize {
                return self.lenFn(self);
            }
            pub fn getTest(self: *Self, idx: usize) TestCase {
                return self.testFn(self, idx);
            }
        };

        pub const TrainingData = struct {
            const Self = @This();
            pub const img_width = 28;
            pub const img_height = 28;
            pub const file_header: [8]u8 = [_]u8{ 'T', 'R', 'A', 'I', 'N', 'D', 'T', 'A' };
            pub const file_version = 0;
            pub const InputImage = @Vector(input_len, Float);

            alloc: *std.mem.Allocator,
            // TODO: unmanaged ArrayLists?
            image_names: std.ArrayList([]u8),
            answers: std.ArrayList(u8),
            images: std.ArrayList(InputImage),
            accessor: TestAccessor,

            pub fn init(alloc: *std.mem.Allocator) Self {
                const images = std.ArrayList(InputImage).init(alloc);
                return .{
                    .alloc = alloc,
                    .image_names = std.ArrayList([]u8).init(alloc),
                    .answers = std.ArrayList(u8).init(alloc),
                    .images = images,
                    .accessor = .{ .testFn = Self.getTest, .lenFn = getLen },
                };
            }

            pub fn load(self: *Self, from_source_only: bool) !void {
                var timer = try std.time.Timer.start();
                try self.loadCsv("./data/digits/train.csv");
                debug.print("CSV loaded {} records in {}\n", .{ self.image_names.items.len, fmtDuration(timer.lap()) });

                if (!from_source_only) so: {
                    self.loadImagesBatch() catch |err| {
                        debug.warn("Could not load images from batch: {}\n", .{err});
                        timer.reset();
                        break :so;
                    };
                    debug.print("Images from batch loaded in: {}\n", .{fmtDuration(timer.lap())});
                    return;
                }
                debug.print("Loading images from sources...\n", .{});
                try self.loadImages("./data/digits/Images/train/");
                debug.print("Images loaded in {}\n", .{fmtDuration(timer.lap())});
            }

            pub fn loadCsv(self: *Self, path: []const u8) !void {
                const file = try std.fs.cwd().openFile(path, .{});
                defer file.close();

                const buffer = try self.alloc.alloc(u8, try file.getEndPos());
                defer self.alloc.free(buffer);
                const csv_tokenizer = &try csv.CsvTokenizer(std.fs.File.Reader).init(file.reader(), buffer, .{});
                const aprox_capacity = buffer.len / 10;
                try self.image_names.ensureTotalCapacity(aprox_capacity);
                try self.answers.ensureTotalCapacity(aprox_capacity);

                const err = error{ CsvExpectsOnlyTwoColumns, CsvMissingColumn };

                { // header
                    _ = try csv_tokenizer.next();
                    _ = try csv_tokenizer.next();
                    _ = try csv_tokenizer.next();
                }

                var i: i32 = 0;
                while (try csv_tokenizer.next()) |first_token| {
                    const name_ptr = try self.image_names.addOne();
                    name_ptr.* = try mem.dupe(self.alloc, u8, first_token.field);
                    if (try csv_tokenizer.next()) |tok| {
                        const digit: u8 = std.fmt.parseInt(u8, tok.field, 10) catch |e| {
                            debug.print("CSV can't parse nubmer: `{s}`", .{first_token.field});
                            return e;
                        };
                        try self.answers.append(digit);
                    } else return err.CsvMissingColumn;

                    if (try csv_tokenizer.next()) |tok| {
                        if (tok != .row_end) return err.CsvExpectsOnlyTwoColumns;
                    }
                    i += 1;
                }
            }

            pub fn loadImages(self: *Self, path_prefix: []const u8) !void {
                const prcent_mod = self.image_names.items.len / 100;
                var path_buff: [512]u8 = undefined;
                mem.copy(u8, path_buff[0..], path_prefix);
                try self.images.resize(self.image_names.items.len);
                for (self.image_names.items) |img_name, i| {
                    if (i % prcent_mod == 0) {
                        debug.print("Image loading progress: {}%\r", .{i * 100 / self.image_names.items.len});
                    }

                    mem.copy(u8, path_buff[path_prefix.len..], img_name[0..]);
                    const path = path_buff[0..(path_prefix.len + img_name.len)];
                    //debug.print("Loading image `{s}`\n", .{path});

                    // read whole file
                    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
                        std.debug.panic("Error opening image `{s}`: {}", .{path, err});
                    };
                    defer file.close();
                    const buffer = try self.alloc.alloc(u8, try file.getEndPos());
                    defer self.alloc.free(buffer);
                    const read_len = try file.readAll(buffer);
                    if (read_len != buffer.len) debug.panic("Read different amount of bytes than in file!", .{});

                    //const map = try fmapper.FMapper.open(path, .{ .mode = .shared }, 0, 0);
                    //defer map.close();
                    //const buffer = map.memory.?;

                    var w: i32 = 0;
                    var h: i32 = 0;
                    var chanells_in_file: i32 = 1;
                    const desired_channels: i32 = 1;
                    const pixels = c.stbi_load_from_memory(buffer.ptr, @intCast(i32, buffer.len), &w, &h, &chanells_in_file, desired_channels);
                    defer std.c.free(pixels);
                    if (pixels == null) {
                        debug.panic("Can't load image {s} filesize: {}", .{ path, buffer.len });
                    }
                    if (desired_channels == chanells_in_file) {
                        debug.panic("Loaded image has different pixel format: conversion not implemented! {}/{}", .{ desired_channels, chanells_in_file });
                    }
                    if (w != img_width or h != img_height) {
                        debug.panic("Wrong image `{s}` size: expected {}x{} got {}x{}", .{ img_name, img_width, img_height, w, h });
                    }

                    for (pixels[0..(img_width * img_height)]) |pixel, i_pix| {
                        self.images.items[i][i_pix] = @intToFloat(Float, pixel) * (1.0 / 255.0);
                    }
                }
            }

            // saves batch training data in single file for faster loading
            pub fn saveImagesBatch(self: *Self) !void {
                const f = try std.fs.cwd().createFile("data/train.data", .{ .truncate = true });
                defer f.close();
                var w = f.writer();
                try w.writeAll(file_header[0..]);
                try w.writeIntLittle(u32, file_version);
                try w.writeIntLittle(u32, img_width);
                try w.writeIntLittle(u32, img_height);

                try w.writeIntLittle(u64, self.images.items.len);
                if (comptime std.Target.current.cpu.arch.endian() != .Little) {
                    @panic("TODO: Implement endian conversion!");
                }
                for (self.images.items) |*img| {
                    try w.writeAll(@ptrCast([*]const u8, img)[0..@sizeOf(@TypeOf(img.*))]);
                }
                debug.warn("Batch saved for future speedup. Run preprocess when source data is changed!\n", .{});
            }

            // loads batched training data
            pub fn loadImagesBatch(self: *Self) !void {
                const f = try std.fs.cwd().openFile("data/train.data", .{});
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
                if (w != img_width or h != img_height) return err.ImageSizeMismatch;

                var records: u64 = try r.readIntLittle(u64);
                try self.images.resize(records);
                if (comptime std.Target.current.cpu.arch.endian() != .Little) {
                    @compileError("TODO: Implement endian conversion!");
                }
                for (self.images.items) |*img| {
                    //const slice = []const u8 { .ptr = @ptrCast([*] const u8, &img.pixels), .len = @typeInfo(img.pixels).size };
                    try r.readNoEof(@ptrCast([*]u8, img)[0..@sizeOf(@TypeOf(img.*))]);
                }
            }

            //  TestAccessor funcs - private
            fn getTest(a: *TestAccessor, idx: usize) TestCase {
                const self = @fieldParentPtr(TrainingData, "accessor", a);
                var answer = @splat(output_len, @as(Float, 0));
                answer[self.answers.items[idx]] = 1.0;
                return .{
                    .name = self.image_names.items[idx],
                    .input = self.images.items[idx],
                    .answer = answer,
                };
            }

            fn getLen(a: *TestAccessor) usize {
                const self = @fieldParentPtr(TrainingData, "accessor", a);
                return self.images.items.len;
            }
        };
    };
}
