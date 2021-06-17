# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transformer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transformer.proto',
  package='',
  syntax='proto3',
  serialized_options=b'H\003',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x11transformer.proto\"\xf8\x02\n\x0c\x45ncoderLayer\x12\x1c\n\x14multihead_norm_scale\x18\x01 \x03(\x02\x12\x1b\n\x13multihead_norm_bias\x18\x02 \x03(\x02\x12$\n\x1cmultihead_project_kernel_qkv\x18\x03 \x03(\x02\x12\"\n\x1amultihead_project_bias_qkv\x18\x04 \x03(\x02\x12\'\n\x1fmultihead_project_kernel_output\x18\x05 \x03(\x02\x12%\n\x1dmultihead_project_bias_output\x18\x06 \x03(\x02\x12\x16\n\x0e\x66\x66n_norm_scale\x18\x07 \x03(\x02\x12\x15\n\rffn_norm_bias\x18\x08 \x03(\x02\x12\x18\n\x10\x66\x66n_first_kernel\x18\t \x03(\x02\x12\x16\n\x0e\x66\x66n_first_bias\x18\n \x03(\x02\x12\x19\n\x11\x66\x66n_second_kernel\x18\x0b \x03(\x02\x12\x17\n\x0f\x66\x66n_second_bias\x18\x0c \x03(\x02\"\x99\x05\n\x0c\x44\x65\x63oderLayer\x12\x17\n\x0fself_norm_scale\x18\x01 \x03(\x02\x12\x16\n\x0eself_norm_bias\x18\x02 \x03(\x02\x12\x1f\n\x17self_project_kernel_qkv\x18\x03 \x03(\x02\x12\x1d\n\x15self_project_bias_qkv\x18\x04 \x03(\x02\x12\"\n\x1aself_project_kernel_output\x18\x05 \x03(\x02\x12 \n\x18self_project_bias_output\x18\x06 \x03(\x02\x12\x19\n\x11\x65ncdec_norm_scale\x18\x07 \x03(\x02\x12\x18\n\x10\x65ncdec_norm_bias\x18\x08 \x03(\x02\x12\x1f\n\x17\x65ncdec_project_kernel_q\x18\t \x03(\x02\x12\x1d\n\x15\x65ncdec_project_bias_q\x18\n \x03(\x02\x12\x1f\n\x17\x65ncdec_project_kernel_k\x18\x0b \x03(\x02\x12\x1d\n\x15\x65ncdec_project_bias_k\x18\x0c \x03(\x02\x12\x1f\n\x17\x65ncdec_project_kernel_v\x18\r \x03(\x02\x12\x1d\n\x15\x65ncdec_project_bias_v\x18\x0e \x03(\x02\x12$\n\x1c\x65ncdec_project_kernel_output\x18\x0f \x03(\x02\x12\"\n\x1a\x65ncdec_project_bias_output\x18\x10 \x03(\x02\x12\x16\n\x0e\x66\x66n_norm_scale\x18\x11 \x03(\x02\x12\x15\n\rffn_norm_bias\x18\x12 \x03(\x02\x12\x18\n\x10\x66\x66n_first_kernel\x18\x13 \x03(\x02\x12\x16\n\x0e\x66\x66n_first_bias\x18\x14 \x03(\x02\x12\x19\n\x11\x66\x66n_second_kernel\x18\x15 \x03(\x02\x12\x17\n\x0f\x66\x66n_second_bias\x18\x16 \x03(\x02\"\xfb\x01\n\x0e\x45mbeddingLayer\x12\x17\n\x0ftoken_embedding\x18\x01 \x03(\x02\x12\x1a\n\x12position_embedding\x18\x02 \x03(\x02\x12\x12\n\nnorm_scale\x18\x03 \x03(\x02\x12\x11\n\tnorm_bias\x18\x04 \x03(\x02\x12\'\n\x1f\x65ncode_output_project_kernel_kv\x18\x05 \x03(\x02\x12%\n\x1d\x65ncode_output_project_bias_kv\x18\x06 \x03(\x02\x12\x13\n\x0bshared_bias\x18\x07 \x03(\x02\x12\x10\n\x08lang_emb\x18\x08 \x03(\x02\x12\x16\n\x0etrg_vocab_mask\x18\t \x03(\x05\"\xcf\x02\n\tModelConf\x12\x10\n\x08head_num\x18\x01 \x01(\x05\x12\x11\n\tbeam_size\x18\x02 \x01(\x05\x12\x1b\n\x13\x65xtra_decode_length\x18\x03 \x01(\x05\x12\x16\n\x0elength_penalty\x18\x04 \x01(\x02\x12\x16\n\x0esrc_padding_id\x18\x05 \x01(\x05\x12\x14\n\x0ctrg_start_id\x18\x06 \x01(\x05\x12\x16\n\x0e\x64iverse_lambda\x18\x07 \x01(\x02\x12\x17\n\x0fsampling_method\x18\x08 \x01(\t\x12\x0c\n\x04topp\x18\t \x01(\x02\x12\x0c\n\x04topk\x18\n \x01(\x05\x12\x12\n\ntrg_end_id\x18\x0b \x01(\x05\x12\x12\n\nis_post_ln\x18\x0c \x01(\x08\x12\x1a\n\x12no_scale_embedding\x18\r \x01(\x08\x12\x10\n\x08use_gelu\x18\x0e \x01(\x08\x12\x17\n\x0fis_multilingual\x18\x0f \x01(\x08\"\xc9\x01\n\x0bTransformer\x12&\n\rsrc_embedding\x18\x01 \x01(\x0b\x32\x0f.EmbeddingLayer\x12$\n\rencoder_stack\x18\x02 \x03(\x0b\x32\r.EncoderLayer\x12&\n\rtrg_embedding\x18\x03 \x01(\x0b\x32\x0f.EmbeddingLayer\x12$\n\rdecoder_stack\x18\x04 \x03(\x0b\x32\r.DecoderLayer\x12\x1e\n\nmodel_conf\x18\x05 \x01(\x0b\x32\n.ModelConfB\x02H\x03\x62\x06proto3'
)




_ENCODERLAYER = _descriptor.Descriptor(
  name='EncoderLayer',
  full_name='EncoderLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='multihead_norm_scale', full_name='EncoderLayer.multihead_norm_scale', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multihead_norm_bias', full_name='EncoderLayer.multihead_norm_bias', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multihead_project_kernel_qkv', full_name='EncoderLayer.multihead_project_kernel_qkv', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multihead_project_bias_qkv', full_name='EncoderLayer.multihead_project_bias_qkv', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multihead_project_kernel_output', full_name='EncoderLayer.multihead_project_kernel_output', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multihead_project_bias_output', full_name='EncoderLayer.multihead_project_bias_output', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_norm_scale', full_name='EncoderLayer.ffn_norm_scale', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_norm_bias', full_name='EncoderLayer.ffn_norm_bias', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_first_kernel', full_name='EncoderLayer.ffn_first_kernel', index=8,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_first_bias', full_name='EncoderLayer.ffn_first_bias', index=9,
      number=10, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_second_kernel', full_name='EncoderLayer.ffn_second_kernel', index=10,
      number=11, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_second_bias', full_name='EncoderLayer.ffn_second_bias', index=11,
      number=12, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=398,
)


_DECODERLAYER = _descriptor.Descriptor(
  name='DecoderLayer',
  full_name='DecoderLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='self_norm_scale', full_name='DecoderLayer.self_norm_scale', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_norm_bias', full_name='DecoderLayer.self_norm_bias', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_project_kernel_qkv', full_name='DecoderLayer.self_project_kernel_qkv', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_project_bias_qkv', full_name='DecoderLayer.self_project_bias_qkv', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_project_kernel_output', full_name='DecoderLayer.self_project_kernel_output', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_project_bias_output', full_name='DecoderLayer.self_project_bias_output', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_norm_scale', full_name='DecoderLayer.encdec_norm_scale', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_norm_bias', full_name='DecoderLayer.encdec_norm_bias', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_kernel_q', full_name='DecoderLayer.encdec_project_kernel_q', index=8,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_bias_q', full_name='DecoderLayer.encdec_project_bias_q', index=9,
      number=10, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_kernel_k', full_name='DecoderLayer.encdec_project_kernel_k', index=10,
      number=11, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_bias_k', full_name='DecoderLayer.encdec_project_bias_k', index=11,
      number=12, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_kernel_v', full_name='DecoderLayer.encdec_project_kernel_v', index=12,
      number=13, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_bias_v', full_name='DecoderLayer.encdec_project_bias_v', index=13,
      number=14, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_kernel_output', full_name='DecoderLayer.encdec_project_kernel_output', index=14,
      number=15, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encdec_project_bias_output', full_name='DecoderLayer.encdec_project_bias_output', index=15,
      number=16, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_norm_scale', full_name='DecoderLayer.ffn_norm_scale', index=16,
      number=17, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_norm_bias', full_name='DecoderLayer.ffn_norm_bias', index=17,
      number=18, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_first_kernel', full_name='DecoderLayer.ffn_first_kernel', index=18,
      number=19, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_first_bias', full_name='DecoderLayer.ffn_first_bias', index=19,
      number=20, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_second_kernel', full_name='DecoderLayer.ffn_second_kernel', index=20,
      number=21, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ffn_second_bias', full_name='DecoderLayer.ffn_second_bias', index=21,
      number=22, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=401,
  serialized_end=1066,
)


_EMBEDDINGLAYER = _descriptor.Descriptor(
  name='EmbeddingLayer',
  full_name='EmbeddingLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='token_embedding', full_name='EmbeddingLayer.token_embedding', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='position_embedding', full_name='EmbeddingLayer.position_embedding', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='norm_scale', full_name='EmbeddingLayer.norm_scale', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='norm_bias', full_name='EmbeddingLayer.norm_bias', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encode_output_project_kernel_kv', full_name='EmbeddingLayer.encode_output_project_kernel_kv', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encode_output_project_bias_kv', full_name='EmbeddingLayer.encode_output_project_bias_kv', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shared_bias', full_name='EmbeddingLayer.shared_bias', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lang_emb', full_name='EmbeddingLayer.lang_emb', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trg_vocab_mask', full_name='EmbeddingLayer.trg_vocab_mask', index=8,
      number=9, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1069,
  serialized_end=1320,
)


_MODELCONF = _descriptor.Descriptor(
  name='ModelConf',
  full_name='ModelConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='head_num', full_name='ModelConf.head_num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='beam_size', full_name='ModelConf.beam_size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='extra_decode_length', full_name='ModelConf.extra_decode_length', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='length_penalty', full_name='ModelConf.length_penalty', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='src_padding_id', full_name='ModelConf.src_padding_id', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trg_start_id', full_name='ModelConf.trg_start_id', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='diverse_lambda', full_name='ModelConf.diverse_lambda', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sampling_method', full_name='ModelConf.sampling_method', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='topp', full_name='ModelConf.topp', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='topk', full_name='ModelConf.topk', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trg_end_id', full_name='ModelConf.trg_end_id', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_post_ln', full_name='ModelConf.is_post_ln', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='no_scale_embedding', full_name='ModelConf.no_scale_embedding', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_gelu', full_name='ModelConf.use_gelu', index=13,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_multilingual', full_name='ModelConf.is_multilingual', index=14,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1323,
  serialized_end=1658,
)


_TRANSFORMER = _descriptor.Descriptor(
  name='Transformer',
  full_name='Transformer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='src_embedding', full_name='Transformer.src_embedding', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encoder_stack', full_name='Transformer.encoder_stack', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trg_embedding', full_name='Transformer.trg_embedding', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='decoder_stack', full_name='Transformer.decoder_stack', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_conf', full_name='Transformer.model_conf', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1661,
  serialized_end=1862,
)

_TRANSFORMER.fields_by_name['src_embedding'].message_type = _EMBEDDINGLAYER
_TRANSFORMER.fields_by_name['encoder_stack'].message_type = _ENCODERLAYER
_TRANSFORMER.fields_by_name['trg_embedding'].message_type = _EMBEDDINGLAYER
_TRANSFORMER.fields_by_name['decoder_stack'].message_type = _DECODERLAYER
_TRANSFORMER.fields_by_name['model_conf'].message_type = _MODELCONF
DESCRIPTOR.message_types_by_name['EncoderLayer'] = _ENCODERLAYER
DESCRIPTOR.message_types_by_name['DecoderLayer'] = _DECODERLAYER
DESCRIPTOR.message_types_by_name['EmbeddingLayer'] = _EMBEDDINGLAYER
DESCRIPTOR.message_types_by_name['ModelConf'] = _MODELCONF
DESCRIPTOR.message_types_by_name['Transformer'] = _TRANSFORMER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EncoderLayer = _reflection.GeneratedProtocolMessageType('EncoderLayer', (_message.Message,), {
  'DESCRIPTOR' : _ENCODERLAYER,
  '__module__' : 'transformer_pb2'
  # @@protoc_insertion_point(class_scope:EncoderLayer)
  })
_sym_db.RegisterMessage(EncoderLayer)

DecoderLayer = _reflection.GeneratedProtocolMessageType('DecoderLayer', (_message.Message,), {
  'DESCRIPTOR' : _DECODERLAYER,
  '__module__' : 'transformer_pb2'
  # @@protoc_insertion_point(class_scope:DecoderLayer)
  })
_sym_db.RegisterMessage(DecoderLayer)

EmbeddingLayer = _reflection.GeneratedProtocolMessageType('EmbeddingLayer', (_message.Message,), {
  'DESCRIPTOR' : _EMBEDDINGLAYER,
  '__module__' : 'transformer_pb2'
  # @@protoc_insertion_point(class_scope:EmbeddingLayer)
  })
_sym_db.RegisterMessage(EmbeddingLayer)

ModelConf = _reflection.GeneratedProtocolMessageType('ModelConf', (_message.Message,), {
  'DESCRIPTOR' : _MODELCONF,
  '__module__' : 'transformer_pb2'
  # @@protoc_insertion_point(class_scope:ModelConf)
  })
_sym_db.RegisterMessage(ModelConf)

Transformer = _reflection.GeneratedProtocolMessageType('Transformer', (_message.Message,), {
  'DESCRIPTOR' : _TRANSFORMER,
  '__module__' : 'transformer_pb2'
  # @@protoc_insertion_point(class_scope:Transformer)
  })
_sym_db.RegisterMessage(Transformer)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
