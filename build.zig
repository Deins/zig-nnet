const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();
    const build_options = b.addOptions();
    build_options.addOption(?[]const u8, "--no-rosegment", null); // for debug symbols to work better: https://github.com/ziglang/zig/issues/1501

    {
        const exe = b.addExecutable("nn", "src/main.zig");

        if (target.isWindows()) {
            exe.want_lto = false; // TODO: remove when fixed. see: https://github.com/ziglang/zig/issues/8531
        }
        exe.addOptions("build_options", build_options);
        exe.addCSourceFile("deps/stb.c", &[_][]const u8{});
        exe.addIncludeDir("deps/stb/");
        exe.addPackagePath("csv", "deps/zig-csv/src/main.zig");
        exe.setTarget(target);
        exe.setBuildMode(mode);
        exe.linkLibC();
        exe.install();

        const run_cmd = exe.run();
        run_cmd.step.dependOn(b.getInstallStep());

        const run_step = b.step("run", "Run the app");
        run_step.dependOn(&run_cmd.step);
    }
}
