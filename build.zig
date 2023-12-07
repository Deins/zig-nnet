const std = @import("std");
const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const optimize = b.standardOptimizeOption(.{});

    const csv_dep = b.dependency("csv", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "nn",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    // For the C-libraries
    exe.linkLibC();

    // TODO: Decide if we need these
    // const build_options = b.addOptions();
    // build_options.addOption(?[]const u8, "--no-rosegment", null); // for debug symbols to work better: https://github.com/ziglang/zig/issues/1501
    // exe.addOptions("build_options", build_options);

    exe.addCSourceFile(.{
        .file = std.build.LazyPath.relative("deps/stb.c"),
        .flags = &[_][]const u8{},
    });
    exe.addIncludePath(.{ .path = "deps/stb/" });
    // Make the `csv` module available to be imported via `@import("csv")`
    exe.addModule("csv", csv_dep.module("zig-csv"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
