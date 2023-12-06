const std = @import("std");
const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

var target: std.zig.CrossTarget = undefined;
var optimize: std.builtin.OptimizeMode = undefined;
fn install(libexe: *LibExeObjStep) void {
    libexe.setTarget(target);
    libexe.setBuildMode(optimize);
    libexe.install();
}

pub fn build(b: *Builder) void {
    target = b.standardTargetOptions(.{});
    optimize = b.standardOptimizeOption(.{});
    const build_options = b.addOptions();
    build_options.addOption(?[]const u8, "--no-rosegment", null); // for debug symbols to work better: https://github.com/ziglang/zig/issues/1501

    const csv_module = b.addModule("csv", .{.source_file = .{.path = "deps/zig-csv/src/main.zig"}});

    {
        const exe = b.addExecutable(.{
            .name = "nn",
            .root_source_file = .{ .path = "src/main.zig" },
            .optimize = optimize,
            .target = target,
        });

        if (target.isWindows()) {
            exe.want_lto = false; // TODO: remove when fixed. see: https://github.com/ziglang/zig/issues/8531
        }
        exe.addOptions("build_options", build_options);
        exe.addCSourceFile(.{ .file = .{ .path = "deps/stb.c" }, .flags = &[_][]const u8{} });
        exe.addIncludePath(.{ .path = "deps/stb/" });
        exe.addModule("csv", csv_module);
        exe.linkLibC();
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());

        const run_step = b.step("run", "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}
