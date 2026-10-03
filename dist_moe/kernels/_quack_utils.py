# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    dsmem_ptr = cute.arch.map_dsmem_ptr(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    )
    return Int32(dsmem_ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    )
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    )
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), (
        "val must be Float32, Int32, or Int64"
    )
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    cute.arch.inline_ptx(
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} "
        "[{$r0}], {$r1}, [{$r2}];",
        read_only_args=[remote_smem_ptr_i32, val, remote_mbar_ptr_i32],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_vector(elem_type, *values, loc=None, ip=None):
    """Build an MLIR vector <N x elem_type> from N scalar DSL values.

    Example: make_vector(cutlass.Uint32, v0, v1) -> <2 x i32> MLIR vector
    """
    from cutlass._mlir import ir

    n = len(values)
    mlir_ty = elem_type.mlir_type
    vec_ty = ir.VectorType.get([n], mlir_ty)
    vec = llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    for i, v in enumerate(values):
        vec = vector.insert(
            elem_type(v).ir_value(loc=loc, ip=ip),
            vec,
            dynamic_position=[],
            static_position=[i],
            loc=loc,
            ip=ip,
        )
    return vec
