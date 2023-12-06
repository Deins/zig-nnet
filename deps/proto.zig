const build = @import("std").build;
const Builder = build.Builder;
const LibExeObjStep = build.LibExeObjStep;

pub fn proto_builder(comptime protobuf_source_dir: []const u8) type {
    return struct {
        // libprotobuf - runtime library
        pub fn addLibProtobuf(libexe : *LibExeObjStep) void {
            libexe.addCSourceFiles(&libprotobuf_files, &.{});
            libexe.addIncludeDir(protobuf_source_dir ++ "/src");
        }

        // libprotoc - proto message compiler library
        pub fn addLibProtoC(libexe : *LibExeObjStep) void {
            libexe.addCSourceFiles(&libprotoc_files, &.{});
            libexe.addIncludeDir(protobuf_source_dir ++ "/src");
        }

        // protoc - protoc message compiler executable
        //  build_libproto - if true builds & links libprotobuf and libprotoc, otherwise have to be linked manually  
        pub fn addExecutableProtoC(b: *Builder, build_libproto : bool) *LibExeObjStep {
            const exe = b.addExecutable("protoc", null);
            exe.addCSourceFile(protobuf_source_dir ++ "/src/google/protobuf/compiler/main.cc", &.{});
            if (build_libproto) {
                addLibProtobuf(exe);
                addLibProtoC(exe);
            }
            exe.addIncludeDir(protobuf_source_dir ++ "/src");
            exe.linkLibCpp();
            return exe;
        }

        // libprotobuf
        const libprotobuf_includes = [_][]const u8{
            protobuf_source_dir ++ "/src/google/protobuf/any.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/api.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/importer.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/parser.h",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor.h",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor_database.h",
            protobuf_source_dir ++ "/src/google/protobuf/duration.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/dynamic_message.h",
            protobuf_source_dir ++ "/src/google/protobuf/empty.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/field_access_listener.h",
            protobuf_source_dir ++ "/src/google/protobuf/field_mask.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/generated_enum_reflection.h",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_bases.h",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_reflection.h",
            protobuf_source_dir ++ "/src/google/protobuf/io/gzip_stream.h",
            protobuf_source_dir ++ "/src/google/protobuf/io/printer.h",
            protobuf_source_dir ++ "/src/google/protobuf/io/tokenizer.h",
            protobuf_source_dir ++ "/src/google/protobuf/map_entry.h",
            protobuf_source_dir ++ "/src/google/protobuf/map_field.h",
            protobuf_source_dir ++ "/src/google/protobuf/map_field_inl.h",
            protobuf_source_dir ++ "/src/google/protobuf/message.h",
            protobuf_source_dir ++ "/src/google/protobuf/metadata.h",
            protobuf_source_dir ++ "/src/google/protobuf/reflection.h",
            protobuf_source_dir ++ "/src/google/protobuf/reflection_ops.h",
            protobuf_source_dir ++ "/src/google/protobuf/service.h",
            protobuf_source_dir ++ "/src/google/protobuf/source_context.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/struct.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/text_format.h",
            protobuf_source_dir ++ "/src/google/protobuf/timestamp.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/type.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/unknown_field_set.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/delimited_message_util.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/field_comparator.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/field_mask_util.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/json_util.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/message_differencer.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/time_util.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/type_resolver.h",
            protobuf_source_dir ++ "/src/google/protobuf/util/type_resolver_util.h",
            protobuf_source_dir ++ "/src/google/protobuf/wire_format.h",
            protobuf_source_dir ++ "/src/google/protobuf/wrappers.pb.h",
        };

        const libprotobuf_files = [_][]const u8{
            protobuf_source_dir ++ "/src/google/protobuf/any.cc",
            protobuf_source_dir ++ "/src/google/protobuf/any.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/api.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/importer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/parser.cc",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor.cc",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/descriptor_database.cc",
            protobuf_source_dir ++ "/src/google/protobuf/duration.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/dynamic_message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/empty.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/extension_set_heavy.cc",
            protobuf_source_dir ++ "/src/google/protobuf/field_mask.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_bases.cc",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_reflection.cc",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_table_driven.cc",
            protobuf_source_dir ++ "/src/google/protobuf/generated_message_tctable_full.cc",
            protobuf_source_dir ++ "/src/google/protobuf/io/gzip_stream.cc",
            protobuf_source_dir ++ "/src/google/protobuf/io/printer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/io/tokenizer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/map_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/reflection_ops.cc",
            protobuf_source_dir ++ "/src/google/protobuf/service.cc",
            protobuf_source_dir ++ "/src/google/protobuf/source_context.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/struct.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/stubs/substitute.cc",
            protobuf_source_dir ++ "/src/google/protobuf/text_format.cc",
            protobuf_source_dir ++ "/src/google/protobuf/timestamp.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/type.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/unknown_field_set.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/delimited_message_util.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/field_comparator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/field_mask_util.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/datapiece.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/default_value_objectwriter.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/error_listener.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/field_mask_utility.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/json_escaping.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/json_objectwriter.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/json_stream_parser.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/object_writer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/proto_writer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/protostream_objectsource.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/protostream_objectwriter.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/type_info.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/internal/utility.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/json_util.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/message_differencer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/time_util.cc",
            protobuf_source_dir ++ "/src/google/protobuf/util/type_resolver_util.cc",
            protobuf_source_dir ++ "/src/google/protobuf/wire_format.cc",
            protobuf_source_dir ++ "/src/google/protobuf/wrappers.pb.cc",
        };

        // libprotoc
        const libprotoc_includes = [_][]const u8{
            protobuf_source_dir ++ "/src/google/protobuf/compiler/code_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/command_line_interface.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_file.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_helpers.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_names.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_doc_comment.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_names.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_options.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/importer.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_kotlin_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_names.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/js/js_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_helpers.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/parser.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/php/php_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/plugin.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/plugin.pb.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/python/python_generator.h",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/ruby/ruby_generator.h",
        };

        const libprotoc_files = [_][]const u8{
            protobuf_source_dir ++ "/src/google/protobuf/compiler/code_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/command_line_interface.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_enum.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_enum_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_extension.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_file.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_helpers.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_map_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_message_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_padding_optimizer.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_parse_function_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_primitive_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_service.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/cpp/cpp_string_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_doc_comment.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_enum.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_enum_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_field_base.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_helpers.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_map_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_message_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_primitive_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_reflection_class.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_repeated_enum_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_repeated_message_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_repeated_primitive_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_source_generator_base.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/csharp/csharp_wrapper_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_context.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_doc_comment.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_enum.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_enum_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_enum_field_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_enum_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_extension.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_extension_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_file.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_generator_factory.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_helpers.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_kotlin_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_map_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_map_field_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message_builder.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message_builder_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message_field_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_message_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_name_resolver.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_primitive_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_primitive_field_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_service.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_shared_code_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_string_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/java/java_string_field_lite.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/js/js_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/js/well_known_types_embed.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_enum.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_enum_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_extension.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_file.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_helpers.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_map_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_message.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_message_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_oneof.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/objectivec/objectivec_primitive_field.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/php/php_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/plugin.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/plugin.pb.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/python/python_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/ruby/ruby_generator.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/subprocess.cc",
            protobuf_source_dir ++ "/src/google/protobuf/compiler/zip_writer.cc",
        };
    };
}
