// most basic binary parsing/serialisation that just adds a header with most basic struct layout validations
// WARNING: doesn't do any endian conversion, or pointer checks/processin (will leed to undefined behaviour)!!!
const std = @import("std");
const Hash = std.crypto.hash.Md5;
const HashVal = [Hash.digest_length]u8;

const Header = packed struct {
    magic : u64 = 0xBF00, // same magic for all bin files, mostly just as format version or for future changes that could potentally keep backwards compatibility if needed 
    type_hash : HashVal = undefined, // hashes comptime type info as validation to check if input/output types match  
    payload_size : u64 = 0,

    fn fromType(comptime t : type) Header {
        return Header{
            .type_hash = comptime structHash(t),
            .payload_size = @sizeOf(t),
        };
    }

    fn matches(self : *@This(), expected: @This()) bool {
        return self.magic == expected.magic and std.mem.eql(u8, self.type_hash[0..], expected.type_hash[0..]) and self.payload_size == expected.payload_size; 
    }
};

const BinFileError =  error {
    TypeMismatch, // hash of member fields
    UnsupportedType,
    FileTooShort,
};

fn structHash(comptime t : type) HashVal {
    var h = Hash.init(.{});
    @setEvalBranchQuota(10000);
    comptime for (@typeInfo(t).Struct.fields) |f| {
        if (f.is_comptime) continue;
        h.update(f.name);
        const alignment : usize = f.alignment;
        h.update(std.mem.asBytes(&alignment));
        const size : usize = @sizeOf(f.field_type);
        h.update(std.mem.asBytes(&size));
        // TODO: more fancy validations, recursively iterate fields, assert on pointers etc. 
        //const fti = @typeInfo(f.field_type);
    };
    var ret : HashVal = undefined;
    h.final(&ret);
    return ret;
}

pub fn writeFile(comptime t : type, v : *t, f : *std.fs.File) !void {
    const h = Header.fromType(t);
    try f.*.writeAll(std.mem.asBytes(&h));
    try f.*.writeAll(std.mem.asBytes(v));
}

pub fn readFile(comptime t : type, v : *t, f : *std.fs.File) !void {
    var h = Header{};
    const h_expected = Header.fromType(t);
    if ((try f.*.readAll(std.mem.asBytes(&h))) != std.mem.asBytes(&h).len)
        return BinFileError.FileTooShort;
    if (!h.matches(h_expected)) return BinFileError.TypeMismatch;
    if ((try f.*.readAll(std.mem.asBytes(v))) != std.mem.asBytes(v).len)
        return BinFileError.FileTooShort;
}