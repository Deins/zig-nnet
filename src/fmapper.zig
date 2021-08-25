const std = @import("std");
const os = std.os;

const windows = struct {
    usingnamespace std.os.windows;
    // docs: https://docs.microsoft.com/en-us/windows/win32/memory/creating-named-shared-memory
    pub extern "kernel32" fn CreateFileMappingW(hFile: HANDLE, lpFileMappingAttributes: [*c]SECURITY_ATTRIBUTES, flProtect: DWORD, dwMaximumSizeHigh: DWORD, dwMaximumSizeLow: DWORD, lpName: [*c]WCHAR) HANDLE;
    pub extern "kernel32" fn CreateFileMappingA(hFile: HANDLE, lpFileMappingAttributes: [*c]SECURITY_ATTRIBUTES, flProtect: DWORD, dwMaximumSizeHigh: DWORD, dwMaximumSizeLow: DWORD, lpName: [*c]CSTR) HANDLE;
    pub extern "kernel32" fn OpenFileMappingA(hFile: HANDLE, lpFileMappingAttributes: LPSECURITY_ATTRIBUTES, flProtect: DWORD, dwMaximumSizeHigh: DWORD, dwMaximumSizeLow: DWORD, lpName: LPCSTR) HANDLE;
    pub extern "kernel32" fn OpenFileMappingW(dwDesiredAccess: DWORD, bInheritHandle: BOOL, lpName: LPCWSTR) HANDLE;
    pub extern "kernel32" fn MapViewOfFile(hFileMappingObject: HANDLE, dwDesiredAccess: DWORD, dwFileOffsetHigh: DWORD, dwFileOffsetLow: DWORD, dwNumberOfBytesToMap: SIZE_T) [*c]align(std.mem.page_size) u8;
    pub extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: LPCVOID) BOOL;
    // OpenFile
    const OF_READ: u32 = 0;
    const OF_READWRITE: u32 = 0x00000002;
    const OF_WRITE = 0x00000001;

    // from wint.h
    const SECTION_QUERY: u32 = 0x0001;
    const SECTION_MAP_WRITE: u32 = 0x0002;
    const SECTION_MAP_READ: u32 = 0x0004;
    const SECTION_MAP_EXECUTE: u32 = 0x0008;
    const SECTION_EXTEND_SIZE: u32 = 0x0010;
    const SECTION_MAP_EXECUTE_EXPLICIT: u32 = 0x0020; // not included in SECTION_ALL_ACCESS
    const SECTION_ALL_ACCESS: u32 = (STANDARD_RIGHTS_REQUIRED | SECTION_QUERY |
        SECTION_MAP_WRITE |
        SECTION_MAP_READ |
        SECTION_MAP_EXECUTE |
        SECTION_EXTEND_SIZE);
    // from memoryapi.h
    const FILE_MAP_WRITE: u32 = SECTION_MAP_WRITE;
    const FILE_MAP_READ: u32 = SECTION_MAP_READ;
    const FILE_MAP_ALL_ACCESS: u32 = SECTION_ALL_ACCESS;
    const FILE_MAP_EXECUTE: u32 = SECTION_MAP_EXECUTE_EXPLICIT; // not included in FILE_MAP_ALL_ACCESS
    const FILE_MAP_COPY: u32 = 0x00000001;
    const FILE_MAP_RESERVE: u32 = 0x80000000;
    const FILE_MAP_TARGETS_INVALID: u32 = 0x40000000;
    const FILE_MAP_LARGE_PAGES: u32 = 0x20000000;
};

pub const FMapper = struct {
    pub const MapError = error{ InavalidFlags, CantMap };
    pub const Cfg = struct {
        // TODO: modes havent been fully tested and might have bugs
        const Mode = enum { 
            private, // copy on write pages
            shared, // shares mmap and file
            exclusive // exclusive file access, fails to open if someone else has opened it. TODO: implement
        };
        read: bool = true,
        write: bool = false,
        //create: bool = false, // creates file if one doesnt exist
        mode: Mode = .shared,
    };

    memory: ?[]align(std.mem.page_size) u8 = null,

    // whole file when len = 0
    pub fn open(path: []const u8, cfg: Cfg, offset: usize, len_: usize) !FMapper {
        var len = len_;
        if (comptime std.Target.current.os.tag == .windows) {
            var flags: u32 = 0;
            var page_flags: u32 = 0;
            var map_flags: u32 = 0;

            if (cfg.read and cfg.write) {
                flags |= windows.OF_READWRITE;
                page_flags |= windows.PAGE_READWRITE;
                map_flags |= windows.FILE_MAP_ALL_ACCESS;
            } else if (cfg.read) {
                flags |= windows.OF_READ;
                page_flags |= windows.FILE_MAP_READ;
                map_flags |= windows.FILE_MAP_READ;
            } else if (cfg.write) {
                flags |= windows.OF_WRITE;
                page_flags |= windows.PAGE_READWRITE;
                map_flags |= windows.FILE_MAP_WRITE;
            } else return MapError.InavalidFlags;

            switch (cfg.mode) {
                Cfg.Mode.private => {
                    page_flags |= windows.PAGE_WRITECOPY;
                    map_flags |= windows.FILE_MAP_COPY;
                },
                Cfg.Mode.shared => {},
                Cfg.Mode.exclusive => {},
            }

            const fd = try os.open(path, 0, 0);
            defer os.close(fd);
            if (len == 0) {
                const f = std.fs.File{ .handle = fd };
                const stat = try f.stat();
                len = stat.size;
            }

            const map_file = windows.CreateFileMappingW(fd, // use paging file
                @intToPtr([*c]windows.SECURITY_ATTRIBUTES, 0), // default security
                page_flags, // read/write access
                @intCast(u32, len >> 32), @intCast(u32, len & 0xFFFFFFFF), null); // name of mapping object
            if (map_file == windows.INVALID_HANDLE_VALUE) {
                const err = std.os.windows.kernel32.GetLastError();
                std.debug.warn("error.Unexpected: windows.CreateFileMapping: GetLastError(): {}\n", .{err});
                return MapError.CantMap;
            }
            defer windows.CloseHandle(map_file);
            const mapped_ptr = windows.MapViewOfFile(map_file, // handle to map object
                windows.FILE_MAP_ALL_ACCESS, // read/write permission
                @intCast(u32, offset >> 32), @intCast(u32, offset & 0xFFFFFFFF), // They must also match the memory allocation granularity of the system TODO: check and report if its invalid
                len);
            if (mapped_ptr == null) {
                const err = std.os.windows.kernel32.GetLastError();
                std.debug.warn("error.Unexpected: windows.MapViewOfFile: GetLastError(): {}\n", .{err});
                return MapError.CantMap;
            }
            const ptr = @ptrCast([*]align(std.mem.page_size) u8, mapped_ptr);
            return FMapper{ .memory = ptr[0..len] };
        }

        { // std.os / posix
            const fd = try os.open(path, flags, mflags.mode);
            defer os.close(fd);
            if (len == 0) {
                const f = std.fs.File{ .handle = fd };
                len = try f.stat().size;
            }
            var prot: u32 = 0;
            if (cfg.write) prot |= os.PROT_WRITE;
            if (cfg.read) prot |= os.PROT_READ;
            var flags: u32 = 0;

            mapped_ptr = os.mmap(null, len, prot, flags, fd, offset);
        }
    }

    pub fn close(self: FMapper) void {
        if (std.Target.current.os.tag == .windows) {
            if (self.memory) |mem| {
                if (windows.UnmapViewOfFile(@ptrCast(windows.LPCVOID, mem.ptr)) == 0) {
                    windows.unexpectedError(std.os.windows.kernel32.GetLastError()) catch return;
                }
                //CloseHandle(self.fd);
            }
            return;
        }

        if (self.memory) |mem| { // std.os / posix
            os.unmap(mem.ptr);
        }
    }
};
