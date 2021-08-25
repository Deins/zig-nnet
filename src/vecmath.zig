@compileError("abandoned prototype playground - unused, untested");
const std = @import("std");

// specialises on type like f32, i32 etc.
pub fn typed(comptime val_type) type {
    return struct {
        // helper for detecting generic struct type  
        const StructType = enum {
            column_vector,
            row_matrix,
            column_matrix,
        };

        // column vector
        pub fn Vector(comptime len : usize) type {
            if (len <= 0) @compileError("Invalid vector length!");
            return struct {
                const Self = @This();
                pub const ValType = val_type;
                pub const Type = StructType.column_vector;
                pub const len = len;
                pub const rows = len;
                pub const cols = 1;

                data : std.meta.Vector(len, ValType) = undefined,
                //data : [len]ValType,

                pub fn add(a : Self, b : Self) Self { 
                    var res = a;
                    res.data *= res.b;
                    return res;
                } 
            };
        }

        // row major matrix
        pub fn Matrix(comptime rows_ : usize, comptime cols_ : usize) type {
            //if (rows_ <= 0 or cols <= 0) @compileError("Matrix can't be of size 0!");
            return struct {
                const Self = @This();
                pub const ValType = val_type;
                pub const Type = StructType.row_matrix;
                pub const rows = rows_;
                pub const cols = rows_;
                pub const len = rows * cols;

                // Matches memory layout of opengl & directx.
                //  matrix:
                //      00 01 02
                //      10 11 12
                // is memory as such: { 00, 01, 02, 10, 11, 12 }
                // therefore it is indexed row major, however see vector is indexed COLUMN major
                rows : [cols]std.meta.Vector(rows, ValType) = undefined,

                pub fn mult(a : Self, b : anytype) Self {
                    comptime if (a.cols != b.rows) 
                        @compileError("Matrix multiplication requires that columns in A equals the number of rows in B!");
                    comptime switch (b.type) {
                        .column_vector => {
                            var result = a;
                            for (rows) |*row| {
                                row.* *= b.data;
                            }
                            return result;
                        },
                        .row_matrix => {
                            // TODO: read, learn, implement : https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf
                            @compileError("Not jet implemented!");
                        },
                    }
                }
            };
        }

        pub fn ColumnMatrix(comptime rows_ : usize, comptime cols_ : usize) type {
            return struct {
                const Self = @This();
                pub const ValType = val_type;
                pub const Type = StructType.row_matrix;
                pub const rows = rows_;
                pub const cols = rows_;
                pub const len = rows * cols;
                cols : [rows]std.meta.Vector(cols, ValType) = undefined,
            };
        }
    };
}


