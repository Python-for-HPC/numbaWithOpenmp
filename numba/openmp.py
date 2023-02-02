from numba.core.withcontexts import WithContext
from lark import Lark, Transformer
from lark.exceptions import VisitError
from numba.core.ir_utils import (
    get_call_table,
    dump_block,
    dump_blocks,
    dprint_func_ir,
    replace_vars,
    apply_copy_propagate_extensions,
    visit_vars_extensions,
    remove_dels,
    visit_vars_inner,
    visit_vars,
    get_name_var_table,
    replace_var_names,
    get_definition,
    build_definitions,
    dead_code_elimination
)
from numba.core.analysis import compute_cfg_from_blocks, compute_use_defs, compute_live_map, find_top_level_loops
from numba.core import ir, config, types, typeinfer, cgutils, compiler, transforms, bytecode, typed_passes, imputils, typing
from numba.core.controlflow import CFGraph
from numba.core.ssa import _run_ssa
from numba.extending import overload
from cffi import FFI
import llvmlite.binding as ll
import llvmlite.llvmpy.core as lc
import llvmlite.ir as lir
import operator
import sys
import copy
import os
import ctypes.util
import numpy as np
from numba.core.analysis import compute_use_defs, compute_live_map, compute_cfg_from_blocks, ir_extension_usedefs, _use_defs_result
import subprocess
import tempfile

library_missing = False

iomplib = os.getenv('NUMBA_OMP_LIB',None)
if iomplib is None:
    iomplib = ctypes.util.find_library("libomp.so")
if iomplib is None:
    iomplib = ctypes.util.find_library("libiomp5.so")
if iomplib is None:
    library_missing = True
else:
    if config.DEBUG_OPENMP >= 1:
        print("Found OpenMP runtime library at", iomplib)
    ll.load_library_permanently(iomplib)

omptargetlib = os.getenv('NUMBA_OMPTARGET_LIB', None)
if omptargetlib is None:
    omptargetlib = ctypes.util.find_library("libomptarget.so")
if omptargetlib is None:
    library_missing = True
else:
    if config.DEBUG_OPENMP >= 1:
        print("Found OpenMP target runtime library at", omptargetlib)
    ll.load_library_permanently(omptargetlib)

#----------------------------------------------------------------------------------------------

class NameSlice:
    def __init__(self, name, the_slice):
        self.name = name
        self.the_slice = the_slice

    def __str__(self):
        return "NameSlice(" + str(self.name) + "," + str(self.the_slice) + ")"


class StringLiteral:
    def __init__(self, x):
        self.x = x


def remove_privatized(x):
    if isinstance(x, ir.Var):
        x = x.name

    if isinstance(x, str) and x.endswith("%privatized"):
        return x[:len(x) - len("%privatized")]
    else:
        return x


def remove_all_privatized(x):
    new_x = None
    while new_x != x:
        new_x = x
        x = remove_privatized(new_x)

    return new_x


def typemap_lookup(typemap, x):
    orig_x = x
    if isinstance(x, ir.Var):
        x = x.name

    while True:
        if x in typemap:
            return typemap[x]
        new_x = remove_privatized(x)
        if new_x == x:
            break
        else:
           x = new_x

    tkeys = typemap.keys()

    # Get basename (without privatized)
    x = remove_all_privatized(x)

    potential_keys = list(filter(lambda y: y.startswith(x), tkeys))

    for pkey in potential_keys:
        pkey_base = remove_all_privatized(pkey)
        if pkey_base == x:
            return typemap[pkey]

    raise KeyError(f"{orig_x} and all of its non-privatized names not found in typemap")


class openmp_tag(object):
    def __init__(self, name, arg=None, load=False, non_arg=False, omp_slice=None):
        self.name = name
        self.arg = arg
        self.load = load
        self.loaded_arg = None
        self.xarginfo = []
        self.non_arg = non_arg
        self.omp_slice = omp_slice

    def var_in(self, var):
        assert isinstance(var, str)

        if isinstance(self.arg, ir.Var):
            return self.arg.name == var

        if isinstance(self.arg, str):
            return self.arg == var

        return False

    def arg_size(self, x, lowerer):
        if config.DEBUG_OPENMP >= 2:
            print("arg_size:", x, type(x))
        if isinstance(x, NameSlice):
            x = x.name
        if isinstance(x, ir.Var):
            # Make sure the var referred to has been alloc'ed already.
            lowerer._alloca_var(x.name, lowerer.fndesc.typemap[x.name])
            if self.load:
                assert(False)
                """
                if not self.loaded_arg:
                    self.loaded_arg = lowerer.loadvar(x.name)
                lop = self.loaded_arg.operands[0]
                loptype = lop.type
                pointee = loptype.pointee
                ref = self.loaded_arg._get_reference()
                decl = str(pointee) + " " + ref
                """
            else:
                arg_str = lowerer.getvar(x.name)
                return lowerer.context.get_abi_sizeof(arg_str.type.pointee)
        elif isinstance(x, lir.instructions.AllocaInstr):
            return lowerer.context.get_abi_sizeof(x.type.pointee)
        elif isinstance(x, str):
            xtyp = lowerer.fndesc.typemap[x]
            if config.DEBUG_OPENMP >= 1:
                print("xtyp:", xtyp, type(xtyp))
            lowerer._alloca_var(x, xtyp)
            if self.load:
                assert(False)
                """
                if not self.loaded_arg:
                    self.loaded_arg = lowerer.loadvar(x)
                lop = self.loaded_arg.operands[0]
                loptype = lop.type
                pointee = loptype.pointee
                ref = self.loaded_arg._get_reference()
                decl = str(pointee) + " " + ref
                """
            else:
                arg_str = lowerer.getvar(x)
                return lowerer.context.get_abi_sizeof(arg_str.type.pointee)
        elif isinstance(x, int):
            assert(False)
            """
            decl = "i32 " + str(x)
            """
        else:
            print("unknown arg type:", x, type(x))
            assert(False)

    def arg_to_str(self, x, lowerer, struct_lower=False, var_table=None):
        if config.DEBUG_OPENMP >= 1:
            print("arg_to_str:", x, type(x), self.load, type(self.load))
        if struct_lower:
            assert isinstance(x, str)
            assert var_table is not None

        typemap = lowerer.fndesc.typemap

        if isinstance(x, NameSlice):
            if config.DEBUG_OPENMP >= 2:
                print("nameslice found:", x)
            x = x.name
        if isinstance(x, ir.Var):
            # Make sure the var referred to has been alloc'ed already.
            lowerer._alloca_var(x.name, typemap_lookup(typemap, x))
            if self.load:
                if not self.loaded_arg:
                    self.loaded_arg = lowerer.loadvar(x.name)
                lop = self.loaded_arg.operands[0]
                loptype = lop.type
                pointee = loptype.pointee
                ref = self.loaded_arg._get_reference()
                decl = str(pointee) + " " + ref
            else:
                arg_str = lowerer.getvar(x.name)
                decl = arg_str.get_decl()
        elif isinstance(x, lir.instructions.AllocaInstr):
            decl = x.get_decl()
        elif isinstance(x, str):
            if "*" in x:
                xsplit = x.split("*")
                assert len(xsplit) == 2
                #xtyp = get_dotted_type(x, typemap, lowerer)
                xtyp = typemap_lookup(typemap, xsplit[0])
                if config.DEBUG_OPENMP >= 1:
                    print("xtyp:", xtyp, type(xtyp))
                lowerer._alloca_var(x, xtyp)
                if self.load:
                    if not self.loaded_arg:
                        self.loaded_arg = lowerer.loadvar(x)
                    lop = self.loaded_arg.operands[0]
                    loptype = lop.type
                    pointee = loptype.pointee
                    ref = self.loaded_arg._get_reference()
                    decl = str(pointee) + " " + ref
                    assert len(xsplit) == 1
                else:
                    arg_str = lowerer.getvar(xsplit[0])
                    #arg_str = lowerer.getvar(x)
                    decl = arg_str.get_decl()
                    if len(xsplit) > 1:
                        cur_typ = xtyp
                        field_indices = []
                        for field in xsplit[1:]:
                            dm = lowerer.context.data_model_manager.lookup(cur_typ)
                            findex = dm._fields.index(field)
                            field_indices.append("i32 " + str(findex))
                            cur_typ = dm._members[findex]
                        fi_str = ",".join(field_indices)
                        decl += f", {fi_str}"
                        #decl = f"SCOPE({decl}, {fi_str})"
            else:
                xtyp = typemap_lookup(typemap, x)
                if config.DEBUG_OPENMP >= 1:
                    print("xtyp:", xtyp, type(xtyp))
                lowerer._alloca_var(x, xtyp)
                if self.load:
                    if not self.loaded_arg:
                        self.loaded_arg = lowerer.loadvar(x)
                    lop = self.loaded_arg.operands[0]
                    loptype = lop.type
                    pointee = loptype.pointee
                    ref = self.loaded_arg._get_reference()
                    decl = str(pointee) + " " + ref
                else:
                    arg_str = lowerer.getvar(x)
                    decl = arg_str.get_decl()

                if struct_lower and isinstance(xtyp, types.npytypes.Array):
                    dm = lowerer.context.data_model_manager.lookup(xtyp)
                    cur_tag_ndim = xtyp.ndim
                    stride_typ = lowerer.context.get_value_type(types.intp) #lc.Type.int(64)
                    stride_abi_size = lowerer.context.get_abi_sizeof(stride_typ)
                    array_var = var_table[self.arg]
                    if config.DEBUG_OPENMP >= 1:
                        print("Found array mapped:", self.name, self.arg, xtyp, type(xtyp), stride_typ, type(stride_typ), stride_abi_size, array_var, type(array_var))
                    size_var = ir.Var(None, self.arg + "_size_var", array_var.loc)
                    #size_var = array_var.scope.redefine("size_var", array_var.loc)
                    size_getattr = ir.Expr.getattr(array_var, "size", array_var.loc)
                    size_assign = ir.Assign(size_getattr, size_var, array_var.loc)
                    typemap[size_var.name] = types.int64
                    lowerer._alloca_var(size_var.name, typemap[size_var.name])
                    lowerer.lower_inst(size_assign)
                    data_field = dm._fields.index("data")
                    shape_field = dm._fields.index("shape")
                    strides_field = dm._fields.index("strides")
                    size_lowered = lowerer.getvar(size_var.name).get_decl()
                    fixed_size = cur_tag_ndim
                    #fixed_size = stride_abi_size * cur_tag_ndim
                    decl += f", i32 {data_field}, i64 0, {size_lowered}"
                    decl += f", i32 {shape_field}, i64 0, i64 {fixed_size}"
                    decl += f", i32 {strides_field}, i64 0, i64 {fixed_size}"

                    # see core/datamodel/models.py
                    #struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*data", non_arg=True, omp_slice=(0,lowerer.loadvar(size_var.name))))
                    #struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*shape", non_arg=True, omp_slice=(0,stride_abi_size * cur_tag_ndim)))
                    #struct_tags.append(openmp_tag(cur_tag.name, cur_tag.arg + "*strides", non_arg=True, omp_slice=(0,stride_abi_size * cur_tag_ndim)))
        elif isinstance(x, StringLiteral):
            decl = str(cgutils.make_bytearray(x.x))
        elif isinstance(x, int):
            decl = "i32 " + str(x)
        else:
            print("unknown arg type:", x, type(x))

        if self.omp_slice is not None:
            def handle_var(x):
                if isinstance(x, ir.Var):
                    loaded_size = lowerer.loadvar(x.name)
                    loaded_op = loaded_size.operands[0]
                    loaded_pointee = loaded_op.type.pointee
                    ret = str(loaded_pointee) + " " + loaded_size._get_reference()
                else:
                    ret = "i64 " + str(x)
                return ret
            start_slice = handle_var(self.omp_slice[0])
            end_slice = handle_var(self.omp_slice[1])
            decl += f", {start_slice}, {end_slice}"
            #decl = f"SLICE({decl}, {self.omp_slice[0]}, {self.omp_slice[1]})"

        return decl

    def post_entry(self, lowerer):
        for xarginfo, xarginfo_args, x, alloca_tuple_list in self.xarginfo:
            loaded_args = [lowerer.builder.load(alloca_tuple[2]) for alloca_tuple in alloca_tuple_list]
            fa_res = xarginfo.from_arguments(lowerer.builder,tuple(loaded_args))
            #fa_res = xarginfo.from_arguments(lowerer.builder,tuple([xarg for xarg in xarginfo_args]))
            assert(len(fa_res) == 1)
            lowerer.storevar(fa_res[0], x)

    def add_length_firstprivate(self, x, lowerer):
        if self.name == "QUAL.OMP.FIRSTPRIVATE":
            return [x]
            #return [x, self.arg_size(x, lowerer)]
            #return [x, lowerer.context.get_constant(types.uintp, self.arg_size(x, lowerer))]
        else:
            return [x]

    def unpack_arg(self, x, lowerer, xarginfo_list):
        if isinstance(x, ir.Var):
            return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, lir.instructions.AllocaInstr):
            return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, str):
            xtyp = lowerer.fndesc.typemap[x]
            if config.DEBUG_OPENMP >= 2:
                print("xtyp:", xtyp, type(xtyp))
            if self.load:
                return self.add_length_firstprivate(x, lowerer), None
            else:
                names_to_unpack = []
                #names_to_unpack = ["QUAL.OMP.FIRSTPRIVATE"]
                #names_to_unpack = ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE"]
                if isinstance(xtyp, types.npytypes.Array) and self.name in names_to_unpack:
                    # from core/datamodel/packer.py
                    xarginfo = lowerer.context.get_arg_packer((xtyp,))
                    xloaded = lowerer.loadvar(x)
                    xarginfo_args = list(xarginfo.as_arguments(lowerer.builder, [xloaded]))
                    xarg_alloca_vars = []
                    for xarg in xarginfo_args:
                        if config.DEBUG_OPENMP >= 2:
                            print("xarg:", type(xarg), xarg, "agg:", xarg.aggregate, type(xarg.aggregate), "ind:", xarg.indices)
                            print(xarg.aggregate.type.elements[xarg.indices[0]])
                        alloca_name = "$alloca_" + xarg.name
                        alloca_typ = xarg.aggregate.type.elements[xarg.indices[0]]
                        alloca_res = lowerer.alloca_lltype(alloca_name, alloca_typ)
                        if config.DEBUG_OPENMP >= 2:
                            print("alloca:", alloca_name, alloca_typ, alloca_res, alloca_res.get_reference())
                        xarg_alloca_vars.append((alloca_name, alloca_typ, alloca_res))
                        lowerer.builder.store(xarg, alloca_res)
                    xarginfo_list.append((xarginfo, xarginfo_args, x, xarg_alloca_vars))
                    rets = []
                    for i, xarg in enumerate(xarg_alloca_vars):
                        rets.append(xarg[2])
                        if i == 4:
                            alloca_name = "$alloca_total_size_" + str(x)
                            if config.DEBUG_OPENMP >= 2:
                                print("alloca_name:", alloca_name)
                            alloca_typ = lowerer.context.get_value_type(types.intp) #lc.Type.int(64)
                            alloca_res = lowerer.alloca_lltype(alloca_name, alloca_typ)
                            if config.DEBUG_OPENMP >= 2:
                                print("alloca:", alloca_name, alloca_typ, alloca_res, alloca_res.get_reference())
                            mul_res = lowerer.builder.mul(lowerer.builder.load(xarg_alloca_vars[2][2]), lowerer.builder.load(xarg_alloca_vars[3][2]))
                            lowerer.builder.store(mul_res, alloca_res)
                            rets.append(alloca_res)
                        else:
                            rets.append(self.arg_size(xarg[2], lowerer))
                    return rets, [x]
                else:
                    return self.add_length_firstprivate(x, lowerer), None
        elif isinstance(x, int):
            return self.add_length_firstprivate(x, lowerer), None
        else:
            print("unknown arg type:", x, type(x))

        return self.add_length_firstprivate(x, lowerer), None

    def unpack_arrays(self, lowerer):
        if isinstance(self.arg, list):
            arg_list = self.arg
        elif self.arg is not None:
            arg_list = [self.arg]
        else:
            return [self]
        new_xarginfo = []
        unpack_res = [self.unpack_arg(arg, lowerer, new_xarginfo) for arg in arg_list]
        new_args = [x[0] for x in unpack_res]
        arrays_to_private = []
        for x in unpack_res:
            if x[1]:
                arrays_to_private.append(x[1])
        ot_res = openmp_tag(self.name, sum(new_args, []), self.load)
        ot_res.xarginfo = new_xarginfo
        return [ot_res] + ([] if len(arrays_to_private) == 0 else [openmp_tag("QUAL.OMP.PRIVATE", sum(arrays_to_private, []), self.load)])

    def lower(self, lowerer, debug):
        builder = lowerer.builder
        decl = ""
        if debug and config.DEBUG_OPENMP >= 1:
            print("openmp_tag::lower", self.name, self.arg, type(self.arg))

        if isinstance(self.arg, list):
            arg_list = self.arg
        elif self.arg is not None:
            arg_list = [self.arg]
        else:
            arg_list = []
        typemap = lowerer.fndesc.typemap
        assert len(arg_list) <= 1

        if self.name == "QUAL.OMP.TARGET.IMPLICIT":
            assert False # shouldn't get here anymore
            if isinstance(typemap[self.arg], types.npytypes.Array):
                name_to_use = "QUAL.OMP.MAP.TOFROM"
            else:
                name_to_use = "QUAL.OMP.FIRSTPRIVATE"
        else:
            name_to_use = self.name

        is_array = self.arg in typemap and isinstance(typemap[self.arg], types.npytypes.Array)

        if name_to_use in ["QUAL.OMP.MAP.TOFROM", "QUAL.OMP.MAP.TO", "QUAL.OMP.MAP.FROM"] and is_array:
            #name_to_use += ".STRUCT"
            #var_table = get_name_var_table(lowerer.func_ir.blocks)
            #decl = ",".join([self.arg_to_str(x, lowerer, struct_lower=True, var_table=var_table) for x in arg_list])
            decl = ",".join([self.arg_to_str(x, lowerer, struct_lower=False) for x in arg_list])
        else:
            decl = ",".join([self.arg_to_str(x, lowerer, struct_lower=False) for x in arg_list])

        return '"' + name_to_use + '"(' + decl + ')'

    def replace_vars_inner(self, var_dict):
        if isinstance(self.arg, ir.Var):
            self.arg = replace_vars_inner(self.arg, var_dict)

    def add_to_use_set(self, use_set):
        if not is_dsa(self.name):
            if isinstance(self.arg, ir.Var):
                use_set.add(self.arg.name)
            if isinstance(self.arg, str):
                use_set.add(self.arg)

    def __str__(self):
        return "openmp_tag(" + str(self.name) + "," + str(self.arg) + ")"


def openmp_tag_list_to_str(tag_list, lowerer, debug):
    tag_strs = [x.lower(lowerer, debug) for x in tag_list]
    return '[ ' + ", ".join(tag_strs) + ' ]'

def add_offload_info(lowerer, new_data):
    if hasattr(lowerer, 'omp_offload'):
        lowerer.omp_offload.append(new_data)
    else:
        lowerer.omp_offload = [new_data]


def get_next_offload_number(lowerer):
    if hasattr(lowerer, 'offload_number'):
        cur = lowerer.offload_number
    else:
        cur = 0
    lowerer.offload_number = cur + 1
    return cur


def list_vars_from_tags(tags):
    used_vars = []
    for t in tags:
        if isinstance(t.arg, ir.Var):
            used_vars.append(t.arg)
    return used_vars


def openmp_region_alloca(obj, alloca_instr, typ):
    obj.alloca(alloca_instr, typ)


def push_alloca_callback(lowerer, callback, data, builder):
    #cgutils.push_alloca_callbacks(callback, data)
    if not hasattr(builder, '_lowerer_push_alloca_callbacks'):
        builder._lowerer_push_alloca_callbacks = 0
    builder._lowerer_push_alloca_callbacks += 1


def pop_alloca_callback(lowerer, builder):
    #cgutils.pop_alloca_callbacks()
    builder._lowerer_push_alloca_callbacks -= 1


def in_openmp_region(builder):
    if hasattr(builder, '_lowerer_push_alloca_callbacks'):
        return builder._lowerer_push_alloca_callbacks > 0
    else:
        return False


def find_target_start_end(func_ir, target_num):
    start_block = None
    end_block = None

    for label, block in func_ir.blocks.items():
        if isinstance(block.body[0], openmp_region_start):
            block_target_num = block.body[0].has_target()
            if target_num == block_target_num:
                start_block = label
                if start_block is not None and end_block is not None:
                    return start_block, end_block
        elif isinstance(block.body[0], openmp_region_end):
            block_target_num = block.body[0].start_region.has_target()
            if target_num == block_target_num:
                end_block = label
                if start_block is not None and end_block is not None:
                    return start_block, end_block

    assert False


def get_tags_of_type(clauses, ctype):
    ret = []
    for c in clauses:
        if c.name == ctype:
            ret.append(c)
    return ret


class openmp_region_start(ir.Stmt):
    def __init__(self, tags, region_number, loc):
        if config.DEBUG_OPENMP >= 2:
            print("region ids openmp_region_start::__init__", id(self))
        self.tags = tags
        self.region_number = region_number
        self.loc = loc
        self.omp_region_var = None
        self.omp_metadata = None
        self.tag_vars = set()
        self.normal_iv = None
        self.target_copy = False
        for tag in self.tags:
            if isinstance(tag.arg, ir.Var):
                self.tag_vars.add(tag.arg.name)
            if isinstance(tag.arg, str):
                self.tag_vars.add(tag.arg)
            if tag.name == "QUAL.OMP.NORMALIZED.IV":
                self.normal_iv = tag.arg
        if config.DEBUG_OPENMP >= 1:
            print("tags:", self.tags)
            print("tag_vars:", self.tag_vars)
        self.acq_res = False
        self.acq_rel = False
        self.alloca_queue = []
        self.end_region = None

    def add_tag(self, tag):
        tag_arg_str = None
        if isinstance(tag.arg, ir.Var):
            tag_arg_str = tag.arg.name
        elif isinstance(tag.arg, str):
            tag_arg_str = tag.arg
        elif isinstance(tag.arg, lir.instructions.AllocaInstr):
            tag_arg_str = tag.arg._get_name()
        else:
            assert False
        if isinstance(tag_arg_str, str):
            self.tag_vars.add(tag_arg_str)
        self.tags.append(tag)

    """
    def __new__(cls, *args, **kwargs):
        instance = super(openmp_region_start, cls).__new__(cls)
        print("openmp_region_start::__new__", id(instance))
        return instance
    """

    """
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        if config.DEBUG_OPENMP >= 2:
            print("region ids openmp_region_start::__copy__", id(self), id(result))
        result.end_region.start_region = result
        return result
    """

    def get_var_dsa(self, var):
        assert isinstance(var, str)
        for tag in self.tags:
            if is_dsa(tag.name) and tag.var_in(var):
                return tag.name
        return None

    def requires_acquire_release(self):
        self.acq_res = True

    def requires_combined_acquire_release(self):
        self.acq_rel = True

    def has_target(self):
        for t in self.tags:
            if t.name == "DIR.OMP.TARGET":
                return t.arg
        return None

    def list_vars(self):
        return list_vars_from_tags(self.tags)

    def update_tags(self):
        with self.builder.goto_block(self.block):
            cur_instr = -1

            while True:
                last_instr = self.builder.block.instructions[cur_instr]
                if isinstance(last_instr, lir.instructions.CallInstr) and last_instr.tags is not None and len(last_instr.tags) > 0:
                    break
                cur_instr -= 1

            last_instr.tags = openmp_tag_list_to_str(self.tags, self.lowerer, False)
            if config.DEBUG_OPENMP >= 1:
                print("last_tags:", last_instr.tags, type(last_instr.tags))

    def alloca(self, alloca_instr, typ):
        # We can't process these right away since the processing required can
        # lead to infinite recursion.  So, we just accumulate them in a queue
        # and then process them later at the end_region marker so that the
        # variables are guaranteed to exist in their full form so that when we
        # process them then they won't lead to infinite recursion.
        self.alloca_queue.append((alloca_instr, typ))

    def process_alloca_queue(self):
        # This should be old code...making sure with the assertion.
        assert len(self.alloca_queue) == 0
        has_update = False
        for alloca_instr, typ in self.alloca_queue:
            has_update = self.process_one_alloca(alloca_instr, typ) or has_update
        if has_update:
            self.update_tags()
        self.alloca_queue = []

    def post_lowering_process_alloca_queue(self, enter_directive):
        has_update = False
        if config.DEBUG_OPENMP >= 1:
            print("starting post_lowering_process_alloca_queue")
        for alloca_instr, typ in self.alloca_queue:
            has_update = self.process_one_alloca(alloca_instr, typ) or has_update
        if has_update:
            if config.DEBUG_OPENMP >= 1:
                print("post_lowering_process_alloca_queue has update:", enter_directive.tags)
            enter_directive.tags = openmp_tag_list_to_str(self.tags, self.lowerer, False)
            # LLVM IR is doing some string caching and the following line is necessary to
            # reset that caching so that the original tag text can be overwritten above.
            enter_directive._clear_string_cache()
            if config.DEBUG_OPENMP >= 1:
                print("post_lowering_process_alloca_queue updated tags:", enter_directive.tags)
        self.alloca_queue = []

    def process_one_alloca(self, alloca_instr, typ):
        avar = alloca_instr.name
        if config.DEBUG_OPENMP >= 1:
            print("openmp_region_start process_one_alloca:", id(self), alloca_instr, avar, typ, type(alloca_instr), self.tag_vars)

        has_update = False
        if self.normal_iv is not None and avar != self.normal_iv and avar.startswith(self.normal_iv):
            for i in range(len(self.tags)):
                if config.DEBUG_OPENMP >= 1:
                    print("Replacing normalized iv with", avar)
                self.tags[i].arg = avar
                has_update = True
                break

        if not self.needs_implicit_vars():
            return has_update
        if avar not in self.tag_vars:
            if config.DEBUG_OPENMP >= 1:
                print(f"LLVM variable {avar} didn't previously exist in the list of vars so adding as private.")
            self.add_tag(openmp_tag("QUAL.OMP.PRIVATE", alloca_instr)) # is FIRSTPRIVATE right here?
            #self.tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", alloca_instr)) # is FIRSTPRIVATE right here?
            has_update = True
        return has_update

    def needs_implicit_vars(self):
        first_tag = self.tags[0]
        if (first_tag.name == "DIR.OMP.PARALLEL" or
            first_tag.name == "DIR.OMP.PARALLEL.LOOP" or
            first_tag.name == "DIR.OMP.TASK"):
            return True
        return False

    def lower(self, lowerer):
        typingctx = lowerer.context.typing_context
        targetctx = lowerer.context
        typemap = lowerer.fndesc.typemap
        calltypes = lowerer.fndesc.calltypes
        context = lowerer.context
        builder = lowerer.builder
        mod = builder.module
        library = lowerer.library
        library.openmp = True
        self.block = builder.block
        self.builder = builder
        self.lowerer = lowerer
        if config.DEBUG_OPENMP >= 1:
            print("region ids lower:block", id(self), self, id(self.block), self.block, type(self.block), self.tags, len(self.tags), "builder_id:", id(self.builder), "block_id:", id(self.block))
            for k,v in lowerer.func_ir.blocks.items():
                print("block post copy:", k, id(v), id(v.body))

        # Convert implicit tags to explicit form now that we have typing info.
        for i in range(len(self.tags)):
            cur_tag = self.tags[i]
            if cur_tag.name == "QUAL.OMP.TARGET.IMPLICIT":
                if isinstance(typemap_lookup(typemap, cur_tag.arg), types.npytypes.Array):
                    cur_tag.name = "QUAL.OMP.MAP.TOFROM"
                else:
                    cur_tag.name = "QUAL.OMP.FIRSTPRIVATE"

        if config.DEBUG_OPENMP >= 1:
            for otag in self.tags:
                print("otag:", otag, type(otag.arg))

        # Remove LLVM vars that might have been added if this is an OpenMP
        # region inside a target region.
        count_alloca_instr = len(list(filter(lambda x: isinstance(x.arg, lir.instructions.AllocaInstr), self.tags)))
        assert(count_alloca_instr == 0)
        #self.tags = list(filter(lambda x: not isinstance(x.arg, lir.instructions.AllocaInstr), self.tags))
        if config.DEBUG_OPENMP >= 1:
            print("after LLVM tag filter", self.tags, len(self.tags))
            for otag in self.tags:
                print("otag:", otag, type(otag.arg))

        """
        tags_unpacked_arrays = []
        for tag in self.tags:
            unpack_res = tag.unpack_arrays(lowerer)
            print("unpack_res:", unpack_res, type(unpack_res))
            tags_unpacked_arrays.extend(unpack_res)
        self.tags = tags_unpacked_arrays
        # get all the array args
        for otag in self.tags:
            print("otag:", otag)
        """

        host_side_target_tags = []
        target_num = self.has_target()
        tgv = None
        if target_num is not None and self.target_copy != True:
            var_table = get_name_var_table(lowerer.func_ir.blocks)

            selected_device = 0
            device_tags = get_tags_of_type(self.tags, "QUAL.OMP.DEVICE")
            if len(device_tags) > 0:
                device_tag = device_tags[-1]
                if isinstance(device_tag.arg, int):
                    selected_device = device_tag.arg
                else:
                    assert False
                print("new selected device:", selected_device)

            extras_before = []
            struct_tags = []
            for i in range(len(self.tags)):
                cur_tag = self.tags[i]
                if cur_tag.name in ["QUAL.OMP.MAP.TOFROM", "QUAL.OMP.MAP.TO", "QUAL.OMP.MAP.FROM"]:
                    assert isinstance(cur_tag.arg, str)
                    cur_tag_typ = typemap_lookup(typemap, cur_tag.arg)
                    if isinstance(cur_tag_typ, types.npytypes.Array):
                        cur_tag_ndim = cur_tag_typ.ndim
                        stride_typ = lowerer.context.get_value_type(types.intp) #lc.Type.int(64)
                        stride_abi_size = context.get_abi_sizeof(stride_typ)
                        array_var = var_table[cur_tag.arg]
                        if config.DEBUG_OPENMP >= 1:
                            print("Found array mapped:", cur_tag.name, cur_tag.arg, cur_tag_typ, type(cur_tag_typ), stride_typ, type(stride_typ), stride_abi_size, array_var, type(array_var))
                        size_var = ir.Var(None, f"{cur_tag.arg}_size_var{target_num}", array_var.loc)
                        #size_var = array_var.scope.redefine("size_var", array_var.loc)
                        size_getattr = ir.Expr.getattr(array_var, "size", array_var.loc)
                        size_assign = ir.Assign(size_getattr, size_var, array_var.loc)
                        typemap[size_var.name] = types.int64
                        lowerer.lower_inst(size_assign)
                        extras_before.append(size_assign)
                        lowerer._alloca_var(size_var.name, typemap[size_var.name])

                        #--------
                        """
                        itemsize_var = ir.Var(None, cur_tag.arg + "_itemsize_var", array_var.loc)
                        itemsize_getattr = ir.Expr.getattr(array_var, "itemsize", array_var.loc)
                        itemsize_assign = ir.Assign(itemsize_getattr, itemsize_var, array_var.loc)
                        typemap[itemsize_var.name] = types.int64
                        lowerer.lower_inst(itemsize_assign)
                        #--------

                        totalsize_var = ir.Var(None, cur_tag.arg + "_totalsize_var", array_var.loc)
                        totalsize_binop = ir.Expr.binop(operator.mul, size_var, itemsize_var, array_var.loc)
                        totalsize_assign = ir.Assign(totalsize_binop, totalsize_var, array_var.loc)
                        calltypes[totalsize_binop] = typing.signature(types.int64, types.int64, types.int64)
                        typemap[totalsize_var.name] = types.int64
                        lowerer.lower_inst(totalsize_assign)
                        #--------
                        """

                        # see core/datamodel/models.py
                        loaded_size = lowerer.loadvar(size_var.name)
                        loaded_op = loaded_size.operands[0]
                        loaded_pointee = loaded_op.type.pointee
                        loaded_str = str(loaded_pointee) + " * " + loaded_size._get_reference()
                        struct_tags.append(openmp_tag(cur_tag.name + ".STRUCT", cur_tag.arg + "*data", non_arg=True, omp_slice=(0, size_var)))
                        #struct_tags.append(openmp_tag(cur_tag.name + ".STRUCT", cur_tag.arg + "*data", non_arg=True, omp_slice=(0, loaded_str)))
                        #struct_tags.append(openmp_tag(cur_tag.name + ".STRUCT", cur_tag.arg + "*data", non_arg=True, omp_slice=(0, lowerer.getvar(size_var.name).get_decl())))
                        #struct_tags.append(openmp_tag(cur_tag.name + ".STRUCT", cur_tag.arg + "*data", non_arg=True, omp_slice=(0, lowerer.loadvar(size_var.name))))
                        struct_tags.append(openmp_tag("QUAL.OMP.MAP.TO.STRUCT", cur_tag.arg + "*shape", non_arg=True, omp_slice=(0, cur_tag_ndim)))
                        struct_tags.append(openmp_tag("QUAL.OMP.MAP.TO.STRUCT", cur_tag.arg + "*strides", non_arg=True, omp_slice=(0, cur_tag_ndim)))
                        cur_tag.name = "QUAL.OMP.MAP.TOFROM"
            self.tags.extend(struct_tags)
            if config.DEBUG_OPENMP >= 1:
                for otag in self.tags:
                    print("tag in target:", otag, type(otag.arg))

            from numba.core.compiler import Compiler, Flags
            #builder.module.device_triples = "spir64"
            if config.DEBUG_OPENMP >= 1:
                print("openmp start region lower has target", type(lowerer.func_ir))
            # Make a copy of the host IR being lowered.
            func_ir = lowerer.func_ir.copy()
            if config.DEBUG_OPENMP >= 1:
                for k,v in lowerer.func_ir.blocks.items():
                    print("region ids block post copy:", k, id(v), id(func_ir.blocks[k]), id(v.body), id(func_ir.blocks[k].body))

            """
            var_table = get_name_var_table(func_ir.blocks)
            new_var_dict = {}
            for name, var in var_table.items():
                new_var_dict[name] = var.scope.redefine(name, var.loc)
            replace_var_names(func_ir.blocks, new_var_dict)
            """

            remove_dels(func_ir.blocks)

            dprint_func_ir(func_ir, "func_ir after remove_dels")

            def fixup_openmp_pairs(blocks):
                """The Numba IR nodes for the start and end of an OpenMP region
                   contain references to each other.  When a target region is
                   outlined that contains these pairs of IR nodes then if we
                   simply shallow copy them then they'll point to their original
                   matching pair in the original IR.  In this function, we go
                   through and find what should be matching pairs in the
                   outlined (target) IR and make those copies point to each
                   other.
                """
                start_dict = {}
                end_dict = {}

                # Go through the blocks in the original IR and create a mapping
                # between the id of the start nodes with their block label and
                # position in the block.  Likewise, do the same for end nodes.
                for label, block in func_ir.blocks.items():
                    for bindex, bstmt in enumerate(block.body):
                        if isinstance(bstmt, openmp_region_start):
                            if config.DEBUG_OPENMP >= 2:
                                print("region ids found region start", id(bstmt))
                            start_dict[id(bstmt)] = (label, bindex)
                        elif isinstance(bstmt, openmp_region_end):
                            if config.DEBUG_OPENMP >= 2:
                                print("region ids found region end", id(bstmt.start_region), id(bstmt))
                            end_dict[id(bstmt.start_region)] = (label, bindex)
                assert(len(start_dict) == len(end_dict))

                # For each start node that we found above, create a copy in the target IR
                # and fixup the references of the copies to point at each other.
                for start_id, blockindex in start_dict.items():
                    start_block, sbindex = blockindex

                    end_block_index = end_dict[start_id]
                    end_block, ebindex = end_block_index

                    if config.DEBUG_OPENMP >= 2:
                        start_pre_copy = blocks[start_block].body[sbindex]
                        end_pre_copy = blocks[end_block].body[ebindex]

                    # Create copy of the OpenMP start and end nodes in the target outlined IR.
                    blocks[start_block].body[sbindex] = copy.copy(blocks[start_block].body[sbindex])
                    blocks[end_block].body[ebindex] = copy.copy(blocks[end_block].body[ebindex])
                    # Reset some fields in the start OpenMP region because the target IR
                    # has not been lowered yet.
                    start_region = blocks[start_block].body[sbindex]
                    start_region.builder = None
                    start_region.block = None
                    start_region.lowerer = None
                    start_region.target_copy = True
                    start_region.tags = copy.deepcopy(start_region.tags)
                    if start_region.has_target() == target_num:
                        start_region.tags.append(openmp_tag("OMP.DEVICE"))
                    end_region = blocks[end_block].body[ebindex]
                    #assert(start_region.omp_region_var is None)
                    assert(len(start_region.alloca_queue) == 0)
                    # Make start and end copies point at each other.
                    end_region.start_region = start_region
                    start_region.end_region = end_region
                    if config.DEBUG_OPENMP >= 2:
                        print(f"region ids fixup start: {id(start_pre_copy)}->{id(start_region)} end: {id(end_pre_copy)}->{id(end_region)}")

            fixup_openmp_pairs(func_ir.blocks)
            state = lowerer.state
            fndesc = lowerer.fndesc
            if config.DEBUG_OPENMP >= 1:
                print("context:", context, type(context))
                print("targetctx:", targetctx, type(targetctx))
                print("state:", state, dir(state))
                print("fndesc:", fndesc, type(fndesc))
                print("func_ir type:", type(func_ir))
            dprint_func_ir(func_ir, "target func_ir")
            internal_codegen = targetctx._internal_codegen
            #target_module = internal_codegen._create_empty_module("openmp.target")

            # Find the start and end IR blocks for this offloaded region.
            start_block, end_block = find_target_start_end(func_ir, target_num)

            remove_openmp_nodes_from_target = False
            if not remove_openmp_nodes_from_target:
                end_target_node = func_ir.blocks[end_block].body[0]

            if config.DEBUG_OPENMP >= 1:
                print("start_block:", start_block)
                print("end_block:", end_block)

            blocks_in_region = get_blocks_between_start_end(func_ir.blocks, start_block, end_block)
            if config.DEBUG_OPENMP >= 1:
                print("blocks_in_region:", blocks_in_region)

            # Find the variables that cross the boundary between the target
            # region and the non-target host-side code.
            ins, outs = transforms.find_region_inout_vars(
                blocks=func_ir.blocks,
                livemap=func_ir.variable_lifetime.livemap,
                callfrom=start_block,
                returnto=end_block,
                body_block_ids=blocks_in_region
            )
            # Get the types of the variables live-in to the target region.
            if config.DEBUG_OPENMP >= 1:
                print("ins:", ins)
                print("outs:", outs)
                print("args:", state.args)
                print("rettype:", state.return_type, type(state.return_type))
            # Re-use Numba loop lifting code to extract the target region as
            # its own function.
            region_info = transforms._loop_lift_info(loop=None,
                                                     inputs=ins,
                                                     outputs=outs,
                                                     callfrom=start_block,
                                                     returnto=end_block)

            region_blocks = dict((k, func_ir.blocks[k]) for k in blocks_in_region)
            transforms._loop_lift_prepare_loop_func(region_info, region_blocks)

            if remove_openmp_nodes_from_target:
                region_blocks[start_block].body = region_blocks[start_block].body[1:]
                #region_blocks[end_block].body = region_blocks[end_block].body[1:]

            #region_blocks = copy.copy(region_blocks)
            #region_blocks = copy.deepcopy(region_blocks)
            # transfer_scope?
            # core/untyped_passes/versioning_loop_bodies

            target_args = []
            outline_arg_typs = []
            for tag in self.tags:
                if config.DEBUG_OPENMP >= 1:
                    print(1, "target_arg?", tag)
                if not tag.non_arg and is_target_arg(tag.name):
                    target_args.append(tag.arg)
                    atyp = get_dotted_type(tag.arg, typemap, lowerer)
                    #atyp = typemap[tag.arg]
                    if is_pointer_target_arg(tag.name, atyp):
                        outline_arg_typs.append(types.CPointer(atyp))
                    else:
                        outline_arg_typs.append(atyp)
                    if config.DEBUG_OPENMP >= 1:
                        print(1, "found target_arg", tag)

            #outline_arg_typs = tuple([typemap[x] for x in target_args])
            if config.DEBUG_OPENMP >= 1:
                print("target_args:", target_args)
                print("outline_arg_typs:", outline_arg_typs)

            # Create the outlined IR from the blocks in the region, making the
            # variables crossing into the regions argument.
            outlined_ir = func_ir.derive(blocks=region_blocks,
                                         arg_names=tuple(target_args),
                                         arg_count=len(target_args),
                                         force_non_generator=True)
            outlined_ir.blocks[start_block].body = extras_before + outlined_ir.blocks[start_block].body
            # Change the name of the outlined function to prepend the
            # word "device" to the function name.
            fparts = outlined_ir.func_id.func_qualname.split('.')
            fparts[-1] = "device" + str(target_num) + fparts[-1]
            outlined_ir.func_id.func_qualname = ".".join(fparts)
            outlined_ir.func_id.func_name = fparts[-1]
            uid = next(bytecode.FunctionIdentity._unique_ids)
            outlined_ir.func_id.unique_name = '{}${}'.format(outlined_ir.func_id.func_qualname, uid)
            if config.DEBUG_OPENMP >= 1:
                print("outlined_ir:", type(outlined_ir), type(outlined_ir.func_id), fparts, outlined_ir.arg_names)
                dprint_func_ir(outlined_ir, "outlined_ir")

            # Create a copy of the state and the typemap inside of it so that changes
            # for compiling the outlined IR don't effect the original compilation state
            # of the host.
            state_copy = copy.copy(state)
            state_copy.typemap = copy.copy(typemap)

            entry_block_num = min(outlined_ir.blocks.keys())
            entry_block = outlined_ir.blocks[entry_block_num]
            if config.DEBUG_OPENMP >= 1:
                print("entry_block:", entry_block)
                for x in entry_block.body:
                    print(x)
            arg_assigns = []
            rev_arg_assigns = []
            cpointer_args = []
            # Add entries in the copied typemap for the arguments to the outlined IR.
            for idx, zipvar in enumerate(zip(target_args, outline_arg_typs)):
                var_in, vtyp = zipvar
                arg_name = "arg." + var_in
                state_copy.typemap.pop(arg_name, None)
                state_copy.typemap[arg_name] = vtyp
                #atyp = typemap[tag.arg]
                #state_copy.typemap[arg_name] = typemap[var_in]
                if isinstance(vtyp, types.CPointer):
                    cpointer_args.append(var_in)
                    arg_assigns.append(ir.Assign(ir.Arg(var_in, idx, self.loc, openmp_ptr=True), ir.Var(None, var_in, self.loc), self.loc))
                    rev_arg_assigns.append(ir.RevArgAssign(ir.Var(None, var_in, self.loc), ir.Arg(var_in, idx, self.loc, openmp_ptr=True, reverse=True), self.loc))

            non_cpointer = []
            # If we are adding a cpointer arg to the entry block then we need to remove
            # any ir.Arg for that variable that the outlining process may have already added.
            for x in entry_block.body[:-1]:
                if isinstance(x, ir.Assign) and isinstance(x.value, ir.Arg):
                    if x.value.name in cpointer_args:
                        continue
                non_cpointer.append(x)
            entry_block.body = non_cpointer + arg_assigns + entry_block.body[-1:]

            last_block = outlined_ir.blocks[end_block]
            if not remove_openmp_nodes_from_target:
                last_block.body = [end_target_node] + last_block.body[:-1] + rev_arg_assigns + last_block.body[-1:]

            assert(isinstance(last_block.body[-1], ir.Return))
            # Add typemap entry for the empty tuple return type.
            state_copy.typemap[last_block.body[-1].value.name] = types.containers.Tuple(())

            if selected_device == 0:
                flags = Flags()

                subtarget = targetctx.subtarget(_registries=dict())
                # Turn off the Numba runtime (incref and decref mostly) for the
                # target compilation.
                subtarget.enable_nrt = False
                printreg = imputils.Registry()
                @printreg.lower(print, types.VarArg(types.Any))
                def print_varargs(context, builder, sig, args):
                    #print("target print_varargs lowerer")
                    return context.get_dummy_value()

                subtarget.install_registry(printreg)
                device_target = subtarget
            elif selected_device == 1:
                from numba.cuda import descriptor as cuda_descriptor, compiler as cuda_compiler
                flags = cuda_compiler.CUDAFlags()
                #cuda_typingctx = cuda_descriptor.cuda_target.typing_context
                #cuda_targetctx = cuda_descriptor.cuda_target.target_context
                device_target = cuda_descriptor.cuda_target.target_context
            else:
                raise NotImplementedError("Unsupported OpenMP device number")

            # Do not compile (generate native code), just lower (to LLVM)
            flags.no_compile = True
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True
            # What to do here?
            flags.forceinline = True
            #flags.fastmath = True
            flags.release_gil = True
            flags.nogil = True
            flags.inline = "always"
            # Create a pipeline that only lowers the outlined target code.  No need to
            # compile because it has already gone through those passes.
            class OnlyLower(compiler.CompilerBase):
                def define_pipelines(self):
                    pms = []
                    if not self.state.flags.force_pyobject:
                        pms.append(compiler.DefaultPassBuilder.define_nopython_lowering_pipeline(self.state))
                    return pms

            if config.DEBUG_OPENMP >= 1:
                print("outlined_ir:", outlined_ir, type(outlined_ir), outlined_ir.arg_names)
                dprint_func_ir(outlined_ir, "outlined_ir")
                dprint_func_ir(func_ir, "target after outline func_ir")
                dprint_func_ir(lowerer.func_ir, "original func_ir")
                print("state_copy.typemap:", state_copy.typemap)
                print("region ids before compile_ir")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")


            cres = compiler.compile_ir(typingctx,
                                       device_target,
                                       outlined_ir,
                                       outline_arg_typs,
                                       #state.args,
                                       types.containers.Tuple(()),  # return types
                                       #types.misc.NoneType('none'),
                                       #state.return_type,
                                       flags,
                                       {},
                                       pipeline_class=OnlyLower,
                                       #pipeline_class=Compiler,
                                       is_lifted_loop=False,  # tried this as True since code derived from loop lifting code but it goes through the pipeline twice and messes things up
                                       parent_state=state_copy)

            if config.DEBUG_OPENMP >= 2:
                print("cres:", type(cres))
                print("fndesc:", cres.fndesc, cres.fndesc.mangled_name)
                print("metadata:", cres.metadata)
            cres_library = cres.library
            if config.DEBUG_OPENMP >= 2:
                print("cres_library:", type(cres_library))
                sys.stdout.flush()
            cres_library._ensure_finalized()
            if config.DEBUG_OPENMP >= 2:
                print("ensure_finalized:")
                sys.stdout.flush()

            if config.DEBUG_OPENMP >= 1:
                print("region ids compile_ir")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")
                print("===================================================================================")

                for k,v in lowerer.func_ir.blocks.items():
                    print("block post copy:", k, id(v), id(func_ir.blocks[k]), id(v.body), id(func_ir.blocks[k].body))

            # TODO: move device pipelines in numba proper.
            if selected_device == 0:
                arch = 'x86_64'
                target_elf = cres_library._get_compiled_object()
                fd_o, filename_o = tempfile.mkstemp('.o')
                with open(filename_o, 'wb') as f:
                    f.write(target_elf)
                fd_so, filename_so = tempfile.mkstemp('.so')
                subprocess.run(['clang', '-shared', filename_o, '-o', filename_so])
                with open(filename_so, 'rb') as f:
                    target_elf = f.read()
                if config.DEBUG_OPENMP >= 1:
                    print('filename_o', filename_o, 'filename_so', filename_so)
                os.close(fd_o)
                os.remove(filename_o)
                os.close(fd_so)
                os.remove(filename_so)

                if config.DEBUG_OPENMP >= 1:
                    print("target_elf:", type(target_elf), len(target_elf))
                    sys.stdout.flush()
            elif selected_device == 1:
                # Explicitly trigger post_lowering_openmp on device code since
                # it is not called by the context.
                post_lowering_openmp(cres_library._module)
                arch = 'nvptx'
                import numba.cuda.api as cudaapi
                cc_api = cudaapi.get_current_device().compute_capability
                cc = 'sm_' + str(cc_api[0]) + str(cc_api[1])
                filename_prefix = cres_library.name
                target_llvm_ir = cres_library.get_llvm_str()
                with open(filename_prefix + '.ll', 'w') as f:
                    f.write(target_llvm_ir)
                subprocess.run(['opt', '-S', '--intrinsics-openmp',
                    filename_prefix + '.ll', '-o', filename_prefix + '-intrinsics_omp.ll'], check=True)
                subprocess.run(['opt', '-S', '-O3', filename_prefix + '-intrinsics_omp.ll',
                    '-o', filename_prefix + '-intrinsics_omp-opt.ll'], check=True)
                omptarget_path = os.path.dirname(omptargetlib)
                libomptarget_arch = omptarget_path + '/libomptarget-' + arch + '-' + cc + '.bc'
                print('libomptarget_arch', libomptarget_arch)
                subprocess.run(['llvm-link', '-S', libomptarget_arch, filename_prefix + '-intrinsics_omp-opt.ll',
                    '-o', filename_prefix + '-intrinsics_omp-opt-linked.ll'], check=True)
                subprocess.run(['clang', '-cc1', '-triple', 'nvptx64-nvidia-cuda',
                    '-target-cpu', cc, '-target-feature', '+ptx64', '-S',
                    filename_prefix + '-intrinsics_omp-opt-linked.ll',
                    '-o', filename_prefix + '-intrinsics_omp-opt-linked.s'], check=True)
                subprocess.run(['ptxas', '-m64', '--gpu-name', cc,
                    filename_prefix + '-intrinsics_omp-opt-linked.s',
                    '-o', filename_prefix + '-intrinsics_omp-opt-linked-opt.o'], check=True)
                with open(filename_prefix + '-intrinsics_omp-opt-linked-opt.o', 'rb') as f:
                    target_elf = f.read()
            else:
                raise NotImplementedError("Unsupported OpenMP device number")

            # if cuda then run ptxas on the cres and pass that

            #bytes_array_typ = lir.ArrayType(cgutils.voidptr_t, len(target_elf))
            #bytes_array_typ = lir.ArrayType(cgutils.int8_t, len(target_elf))
            #dev_image = cgutils.add_global_variable(mod, bytes_array_typ, ".omp_offloading.device_image")
            #dev_image.initializer = lc.Constant.array(cgutils.int8_t, target_elf)
            #dev_image.initializer = lc.Constant.array(cgutils.int8_t, target_elf)
            add_target_globals_in_numba = int(os.environ.get("NUMBA_OPENMP_ADD_TARGET_GLOBALS", 0))
            if add_target_globals_in_numba != 0:
                elftext = cgutils.make_bytearray(target_elf)
                dev_image = targetctx.insert_unique_const(mod, ".omp_offloading.device_image", elftext)
                mangled_name = cgutils.make_bytearray(cres.fndesc.mangled_name.encode("utf-8") + b"\x00")
                mangled_var = targetctx.insert_unique_const(mod, ".omp_offloading.entry_name", mangled_name)

                llvmused_typ = lir.ArrayType(cgutils.voidptr_t, 2)
                llvmused_gv = cgutils.add_global_variable(mod, llvmused_typ, "llvm.used")
                llvmused_syms = [lc.Constant.bitcast(dev_image, cgutils.voidptr_t),
                                 lc.Constant.bitcast(mangled_var, cgutils.voidptr_t)]
                llvmused_gv.initializer = lc.Constant.array(cgutils.voidptr_t, llvmused_syms)
                llvmused_gv.linkage = "appending"
            else:
                host_side_target_tags.append(openmp_tag("QUAL.OMP.TARGET.DEV_FUNC", StringLiteral(cres.fndesc.mangled_name.encode("utf-8"))))
                host_side_target_tags.append(openmp_tag("QUAL.OMP.TARGET.ELF", StringLiteral(target_elf)))

            """
            llvmused_typ = lir.ArrayType(cgutils.voidptr_t, 1)
            llvmused_gv = cgutils.add_global_variable(mod, llvmused_typ, "llvm.used")
            llvmused_syms = [lc.Constant.bitcast(dev_image, cgutils.voidptr_t)]
            llvmused_gv.initializer = lc.Constant.array(cgutils.voidptr_t, llvmused_syms)
            llvmused_gv.linkage = "appending"
            """

            """
            todd_gv1_typ = targetctx.get_value_type(types.intp) #lc.Type.int(64)
            tgv = cgutils.add_global_variable(target_module, todd_gv1_typ, "todd_gv1")
            tst = targetctx.insert_const_string(target_module, "todd_string2")
            llvmused_typ = lir.ArrayType(cgutils.voidptr_t, 1)
            #llvmused_typ = lir.ArrayType(cgutils.voidptr_t, 2)
            #llvmused_typ = lir.ArrayType(cgutils.intp_t, 2)
            llvmused_gv = cgutils.add_global_variable(target_module, llvmused_typ, "llvm.used")
            llvmused_syms = [lc.Constant.bitcast(tgv, cgutils.voidptr_t)]
            #llvmused_syms = [tgv, tst]
            llvmused_gv.initializer = lc.Constant.array(cgutils.voidptr_t, llvmused_syms)
            #llvmused_gv.initializer = lc.Constant.array(cgutils.intp_t, llvmused_syms)
            #llvmused_gv.initializer = lc.Constant.array(llvmused_typ, llvmused_syms)
            llvmused_gv.linkage = "appending"

            #targetctx.declare_function(target_module, )
            library.add_ir_module(target_module)
            """

            if config.DEBUG_OPENMP >= 1:
                dprint_func_ir(func_ir, "target after outline compiled func_ir")

        llvm_token_t = lc.Type.token()
        fnty = lir.FunctionType(llvm_token_t, [])
        tags_to_include = self.tags + host_side_target_tags
        #tags_to_include = list(filter(lambda x: x.name != "DIR.OMP.TARGET", tags_to_include))
        self.filtered_tag_length = len(tags_to_include)
        if config.DEBUG_OPENMP >= 1:
            print("filtered_tag_length:", self.filtered_tag_length)
        #print("FIX FIX FIX....this works during testing but not in target is combined with other options.  We need to remove all the target related options and then if nothing is left we can skip adding this region.")
        if len(tags_to_include) > 0:
            if config.DEBUG_OPENMP >= 1:
                print("push_alloca_callbacks")

            push_alloca_callback(lowerer, openmp_region_alloca, self, builder)
            tag_str = openmp_tag_list_to_str(tags_to_include, lowerer, True)
            pre_fn = builder.module.declare_intrinsic('llvm.directive.region.entry', (), fnty)
            assert(self.omp_region_var is None)
            self.omp_region_var = builder.call(pre_fn, [], tail=False, tags=tag_str)
            # This is used by the post-lowering pass over LLVM to add LLVM alloca
            # vars to the Numba IR openmp node and then when the exit of the region
            # is detected then the tags in the enter directive are updated.
            self.omp_region_var.save_orig_numba_openmp = self
            if config.DEBUG_OPENMP >= 2:
                print("setting omp_region_var", self.omp_region_var._get_name())
        """
        if self.omp_metadata is None and self.has_target():
            self.omp_metadata = builder.module.add_metadata([
                 lir.IntType(32)(0),   # Kind of this metadata.  0 is for target.
                 lir.IntType(32)(lb.getDeviceForFile(self.loc.filename)),   # Device ID of the file with the entry.
                 lir.IntType(32)(lb.getFileIdForFile(self.loc.filename)),   # File ID of the file with the entry.
                 lowerer.fndesc.mangled_name,   # Mangled name of the function with the entry.
                 lir.IntType(32)(self.loc.line),  # Line in the source file where with the entry.
                 lir.IntType(32)(self.region_number),   # Order the entry was created.
                 #lir.IntType(32)(get_next_offload_number(lowerer)),   # Order the entry was created.
                 lir.IntType(32)(0)    # Entry kind.  Should always be 0 I think.
                ])
            add_offload_info(lowerer, self.omp_metadata)
        """
        if self.acq_res:
            builder.fence("acquire")
        if self.acq_rel:
            builder.fence("acq_rel")

        for otag in self.tags:  # should be tags_to_include?
            otag.post_entry(lowerer)

        if config.DEBUG_OPENMP >= 1:
            sys.stdout.flush()

    def __str__(self):
        return "openmp_region_start " + ", ".join([str(x) for x in self.tags]) + " target=" + str(self.target_copy)


class openmp_region_end(ir.Stmt):
    def __init__(self, start_region, tags, loc):
        if config.DEBUG_OPENMP >= 1:
            print("region ids openmp_region_end::__init__", id(self), id(start_region))
        self.start_region = start_region
        self.tags = tags
        self.loc = loc
        self.start_region.end_region = self

    def __new__(cls, *args, **kwargs):
        instance = super(openmp_region_end, cls).__new__(cls)
        #print("openmp_region_end::__new__", id(instance))
        return instance

    def list_vars(self):
        return list_vars_from_tags(self.tags)

    def lower(self, lowerer):
        typingctx = lowerer.context.typing_context
        targetctx = lowerer.context
        typemap = lowerer.fndesc.typemap
        context = lowerer.context
        builder = lowerer.builder
        library = lowerer.library

        if config.DEBUG_OPENMP >= 2:
            print("openmp_region_end::lower", id(self), id(self.start_region))
            sys.stdout.flush()

        if self.start_region.acq_res:
            builder.fence("release")

        if config.DEBUG_OPENMP >= 1:
            print("pop_alloca_callbacks")

        if config.DEBUG_OPENMP >= 2:
            print("start_region tag length:", self.start_region.filtered_tag_length)

        if self.start_region.filtered_tag_length > 0:
            llvm_token_t = lc.Type.token()
            fnty = lir.FunctionType(lc.Type.void(), [llvm_token_t])
            # The callback is only needed if llvm.directive.region.entry was added
            # which only happens if tag length > 0.
            pop_alloca_callback(lowerer, builder)

            # Process the accumulated allocas in the start region.
            self.start_region.process_alloca_queue()

            assert self.start_region.omp_region_var != None
            if config.DEBUG_OPENMP >= 2:
                print("before adding exit", self.start_region.omp_region_var._get_name())
            pre_fn = builder.module.declare_intrinsic('llvm.directive.region.exit', (), fnty)
            builder.call(pre_fn, [self.start_region.omp_region_var], tail=True, tags=openmp_tag_list_to_str(self.tags, lowerer, True))

    def __str__(self):
        return "openmp_region_end " + ", ".join([str(x) for x in self.tags])

    def has_target(self):
        for t in self.tags:
            if t.name == "DIR.OMP.TARGET":
                return t.arg
        return None


def compute_cfg_from_llvm_blocks(blocks):
    cfg = CFGraph()
    name_to_index = {}
    for b in blocks:
        #print("b:", b.name, type(b.name))
        cfg.add_node(b.name)

    for bindex, b in enumerate(blocks):
        term = b.terminator
        #print("term:", b.name, term, type(term))
        if isinstance(term, lir.instructions.Branch):
            cfg.add_edge(b.name, term.operands[0].name)
            name_to_index[b.name] = (bindex, [term.operands[0].name])
        elif isinstance(term, lir.instructions.ConditionalBranch):
            cfg.add_edge(b.name, term.operands[1].name)
            cfg.add_edge(b.name, term.operands[2].name)
            name_to_index[b.name] = (bindex, [term.operands[1].name, term.operands[2].name])
        elif isinstance(term, lir.instructions.Ret):
            name_to_index[b.name] = (bindex, [])
        elif isinstance(term, lir.instructions.SwitchInstr):
            cfg.add_edge(b.name, term.default.name)
            for _, blk in term.cases:
                cfg.add_edge(b.name, blk.name)
            out_blks = [x[1].name for x in term.cases]
            out_blks.append(term.default.name)
            name_to_index[b.name] = (bindex, out_blks)
        else:
            print("Unknown term:", term, type(term))
            assert(False) # Should never get here.

    cfg.set_entry_point("entry")
    cfg.process()
    return cfg, name_to_index


def compute_llvm_topo_order(blocks):
    cfg, name_to_index = compute_cfg_from_llvm_blocks(blocks)
    post_order = []
    seen = set()

    def _dfs_rec(node):
        if node not in seen:
            seen.add(node)
            succs = cfg._succs[node]

            # This is needed so that the inside of loops are
            # handled first before their exits.
            nexts = name_to_index[node][1]
            if len(nexts) == 2:
                succs = [nexts[1], nexts[0]]

            for dest in succs:
                if (node, dest) not in cfg._back_edges:
                    _dfs_rec(dest)
            post_order.append(node)

    _dfs_rec(cfg.entry_point())
    post_order.reverse()
    return post_order, name_to_index


class CollectUnknownLLVMVarsPrivate(lir.transforms.Visitor):
    def __init__(self):
        self.active_openmp_directives = []
        self.start_num = 0

    # Override the default function visitor to go in topo order
    def visit_Function(self, func):
        self._function = func
        if len(func.blocks) == 0:
            return None
        if config.DEBUG_OPENMP >= 1:
            print("Collect visit_Function:", func.blocks, type(func.blocks))
        topo_order, name_to_index = compute_llvm_topo_order(func.blocks)
        topo_order = list(topo_order)
        if config.DEBUG_OPENMP >= 1:
            print("topo_order:", topo_order)

        for bbname in topo_order:
            self.visit_BasicBlock(func.blocks[name_to_index[bbname][0]])

        if config.DEBUG_OPENMP >= 1:
            print("Collect visit_Function done")

    def visit_Instruction(self, instr):
        if len(self.active_openmp_directives) > 0:
            if config.DEBUG_OPENMP >= 1:
                print("Collect instr:", instr, type(instr))
            for op in instr.operands:
                if isinstance(op, lir.AllocaInstr):
                    if config.DEBUG_OPENMP >= 1:
                        print("Collect AllocaInstr operand:", op, op.name)
                    for directive in self.active_openmp_directives:
                        directive.save_orig_numba_openmp.alloca(op, None)
                else:
                    if config.DEBUG_OPENMP >= 2:
                        print("non-alloca:", op, type(op))
                    pass

        if isinstance(instr, lir.CallInstr):
            if instr.callee.name == 'llvm.directive.region.entry':
                if config.DEBUG_OPENMP >= 1:
                    print("Collect Found openmp region entry:", instr, type(instr), "\n", instr.tags, type(instr.tags))
                self.active_openmp_directives.append(instr)
                assert hasattr(instr, "save_orig_numba_openmp")
            if instr.callee.name == 'llvm.directive.region.exit':
                if config.DEBUG_OPENMP >= 1:
                    print("Collect Found openmp region exit:", instr, type(instr), "\n", instr.tags, type(instr.tags))
                enter_directive = self.active_openmp_directives.pop()
                enter_directive.save_orig_numba_openmp.post_lowering_process_alloca_queue(enter_directive)


def post_lowering_openmp(mod):
    if config.DEBUG_OPENMP >= 1:
        print("post_lowering_openmp")

    # This will gather the information.
    collect_fixup = CollectUnknownLLVMVarsPrivate()
    collect_fixup.visit(mod)

    if config.DEBUG_OPENMP >= 1:
        print("post_lowering_openmp done")

# Callback for ir_extension_usedefs
def openmp_region_start_defs(region, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    for tag in region.tags:
        tag.add_to_use_set(use_set)
    return _use_defs_result(usemap=use_set, defmap=def_set)

def openmp_region_end_defs(region, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
    for tag in region.tags:
        tag.add_to_use_set(use_set)
    return _use_defs_result(usemap=use_set, defmap=def_set)

# Extend usedef analysis to support openmp_region_start/end nodes.
ir_extension_usedefs[openmp_region_start] = openmp_region_start_defs
ir_extension_usedefs[openmp_region_end] = openmp_region_end_defs

def openmp_region_start_infer(prs, typeinferer):
    pass

def openmp_region_end_infer(pre, typeinferer):
    pass

typeinfer.typeinfer_extensions[openmp_region_start] = openmp_region_start_infer
typeinfer.typeinfer_extensions[openmp_region_end] = openmp_region_end_infer

def _lower_openmp_region_start(lowerer, prs):
    prs.lower(lowerer)

def _lower_openmp_region_end(lowerer, pre):
    pre.lower(lowerer)

def apply_copies_openmp_region(region, var_dict, name_var_table, typemap, calltypes, save_copies):
    for i in range(len(region.tags)):
        region.tags[i].replace_vars_inner(var_dict)

apply_copy_propagate_extensions[openmp_region_start] = apply_copies_openmp_region
apply_copy_propagate_extensions[openmp_region_end] = apply_copies_openmp_region

def visit_vars_openmp_region(region, callback, cbdata):
    for i in range(len(region.tags)):
        if config.DEBUG_OPENMP >= 1:
            print("visit_vars before", region.tags[i], type(region.tags[i].arg))
        region.tags[i].arg = visit_vars_inner(region.tags[i].arg, callback, cbdata)
        if config.DEBUG_OPENMP >= 1:
            print("visit_vars after", region.tags[i])

visit_vars_extensions[openmp_region_start] = visit_vars_openmp_region
visit_vars_extensions[openmp_region_end] = visit_vars_openmp_region

#----------------------------------------------------------------------------------------------

class PythonOpenmp:
    def __init__(self, *args):
        self.args = args

    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass


def extract_args_from_openmp(func_ir):
    """ Find all the openmp context calls in the function and then
        use the VarCollector transformer to find all the Python variables
        referenced in the openmp clauses.  We then add those variables as
        regular arguments to the openmp context call just so Numba's
        usedef analysis is able to keep variables alive that are only
        referenced in openmp clauses.
    """
    func_ir._definitions = build_definitions(func_ir.blocks)
    var_table = get_name_var_table(func_ir.blocks)
    for block in func_ir.blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and inst.value.op == "call":
                func_def = get_definition(func_ir, inst.value.func)
                if isinstance(func_def, ir.Global) and isinstance(func_def.value, _OpenmpContextType):
                    str_def = get_definition(func_ir, inst.value.args[0])
                    if not isinstance(str_def, ir.Const) or not isinstance(str_def.value, str):
                        # The non-const openmp string error is handled later.
                        continue
                    assert isinstance(str_def, ir.Const) and isinstance(str_def.value, str)
                    parse_res = var_collector_parser.parse(str_def.value)
                    visitor = VarCollector()
                    try:
                        visit_res = visitor.transform(parse_res)
                        inst.value.args.extend([var_table[x] for x in visit_res])
                    except Exception as f:
                        print("generic transform exception")
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        #print("Internal error for OpenMp pragma '{}'".format(arg.value))
                        sys.exit(-2)
                    except:
                        print("fallthrough exception")
                        #print("Internal error for OpenMp pragma '{}'".format(arg.value))
                        sys.exit(-3)


class _OpenmpContextType(WithContext):
    is_callable = True

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra, state=None, flags=None):
        if config.DEBUG_OPENMP >= 1:
            print("pre-dead-code")
            dump_blocks(blocks)
        if not config.OPENMP_DISABLED and not hasattr(func_ir, "has_openmp_region"):
            # We can't do dead code elimination at this point because if an argument
            # is used only in an openmp clause then it is detected as dead and is
            # eliminated.  We'd have to run through the IR and find all the
            # openmp regions and extract the vars used there and then modify the
            # IR with something fake just to take the var alive.  The other approach
            # would be to modify dead code elimination to find the vars referenced
            # in openmp context strings.
            extract_args_from_openmp(func_ir)
            #dead_code_elimination(func_ir)
            remove_ssa_from_func_ir(func_ir)
            func_ir.has_openmp_region = True
        if config.DEBUG_OPENMP >= 1:
            print("pre-with-removal")
            dump_blocks(blocks)
        if config.OPENMP_DISABLED:
            # If OpenMP disabled, do nothing except remove the enter_with marker.
            sblk = blocks[blk_start]
            sblk.body = sblk.body[1:]
        else:
            if config.DEBUG_OPENMP >= 1:
                print("openmp:mutate_with_body")
                dprint_func_ir(func_ir, "func_ir")
                print("blocks:", blocks, type(blocks))
                print("blk_start:", blk_start, type(blk_start))
                print("blk_end:", blk_end, type(blk_end))
                print("body_blocks:", body_blocks, type(body_blocks))
                print("extra:", extra, type(extra))
                print("flags:", flags, type(flags))
            assert extra is not None
            assert flags is not None
            flags.enable_ssa = False
            flags.release_gil = True
            flags.noalias = True
            _add_openmp_ir_nodes(func_ir, blocks, blk_start, blk_end, body_blocks, extra, state)
            func_ir._definitions = build_definitions(func_ir.blocks)
            if config.DEBUG_OPENMP >= 1:
                print("post-with-removal")
                dump_blocks(blocks)
            dispatcher = dispatcher_factory(func_ir)
            dispatcher.can_cache = True
            return dispatcher

    def __call__(self, args):
        return PythonOpenmp(args)


def remove_indirections(clause):
    try:
        while len(clause) == 1 and isinstance(clause[0], list):
            clause = clause[0]
    except:
        pass
    return clause


class default_shared_val:
    def __init__(self, val):
        self.val = val


class UnspecifiedVarInDefaultNone(Exception):
    pass

class ParallelForExtraCode(Exception):
    pass

class ParallelForWrongLoopCount(Exception):
    pass

class NonconstantOpenmpSpecification(Exception):
    pass

class NonStringOpenmpSpecification(Exception):
    pass

class MultipleNumThreadsClauses(Exception):
    pass

openmp_context = _OpenmpContextType()


def is_dsa(name):
    return name in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.PRIVATE", "QUAL.OMP.SHARED", "QUAL.OMP.LASTPRIVATE", "QUAL.OMP.TARGET.IMPLICIT"] or name.startswith("QUAL.OMP.REDUCTION") or name.startswith("QUAL.OMP.MAP")


def get_dotted_type(x, typemap, lowerer):
    xsplit = x.split("*")
    cur_typ = typemap_lookup(typemap, xsplit[0])
    #print("xsplit:", xsplit, cur_typ, type(cur_typ))
    for field in xsplit[1:]:
        dm = lowerer.context.data_model_manager.lookup(cur_typ)
        findex = dm._fields.index(field)
        cur_typ = dm._members[findex]
        #print("dm:", dm, type(dm), dm._members, type(dm._members), dm._fields, type(dm._fields), findex, cur_typ, type(cur_typ))
    return cur_typ


def is_target_arg(name):
    return name in ["QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.TARGET.IMPLICIT"] or name.startswith("QUAL.OMP.MAP")


def is_pointer_target_arg(name, typ):
    if name.startswith("QUAL.OMP.MAP"):
        return True
    if name in ["QUAL.OMP.FIRSTPRIVATE"]:
        return False
    if name in ["QUAL.OMP.TARGET.IMPLICIT"]:
        if isinstance(typ, types.npytypes.Array):
            return True
        else:
            return False
    assert False


def is_internal_var(var):
    # Determine if a var is a Python var or an internal Numba var.
    if var.is_temp:
        return True
    return var.unversioned_name != var.name


def remove_ssa(var_name, scope, loc):
    # Get the base name of a variable, removing the SSA extension.
    var = ir.Var(scope, var_name, loc)
    return var.unversioned_name


def user_defined_var(var):
    if not isinstance(var, str):
        return False
    return not var.startswith("$")


def has_user_defined_var(the_set):
    for x in the_set:
        if user_defined_var(x):
            return True
    return False


def get_user_defined_var(the_set):
    ret = set()
    for x in the_set:
        if user_defined_var(x):
            ret.add(x)
    return ret


unique = 0
def get_unique():
    global unique
    ret = unique
    unique += 1
    return ret


def is_private(x):
    return x in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.LASTPRIVATE", "QUAL.OMP.TARGET.IMPLICIT"]


def openmp_copy(a):
    pass  # should always be called through overload


@overload(openmp_copy)
def openmp_copy_overload(a):
    if config.DEBUG_OPENMP >= 1:
        print("openmp_copy:", a, type(a))
    if isinstance(a, types.npytypes.Array):
        def cimpl(a):
            return np.copy(a)
        return cimpl
    else:
        def cimpl(a):
            return a
        return cimpl


def replace_ssa_var_callback(var, vardict):
    assert isinstance(var, ir.Var)
    while var.unversioned_name in vardict.keys():
        assert(vardict[var.unversioned_name].name != var.unversioned_name)
        new_var = vardict[var.unversioned_name]
        var = ir.Var(new_var.scope, new_var.name, new_var.loc)
    return var


def replace_ssa_vars(blocks, vardict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> ir.Var)"""
    # remove identity values to avoid infinite loop
    new_vardict = {}
    for l, r in vardict.items():
        if l != r.name:
            new_vardict[l] = r
    visit_vars(blocks, replace_ssa_var_callback, new_vardict)


def get_blocks_between_start_end(blocks, start_block, end_block):
    cfg = compute_cfg_from_blocks(blocks)
    blocks_in_region = [start_block]
    def add_in_region(cfg, blk, blocks_in_region, end_block):
        """For each successor in the CFG of the block we're currently
           adding to blocks_in_region, add that successor to
           blocks_in_region if it isn't the end_block.  Then,
           recursively call this routine for the added block to add
           its successors.
        """
        for out_blk, _ in cfg.successors(blk):
            if out_blk != end_block and out_blk not in blocks_in_region:
                blocks_in_region.append(out_blk)
                add_in_region(cfg, out_blk, blocks_in_region, end_block)

    # Calculate all the Numba IR blocks in the target region.
    add_in_region(cfg, start_block, blocks_in_region, end_block)
    return blocks_in_region

class VarName(str):
    pass

class OnlyClauseVar(VarName):
    pass

# This Transformer visitor class just finds the referenced python names
# and puts them in a list of VarName.  The default visitor function
# looks for list of VarNames in the args to that tree node and then
# concatenates them all together.  The final return value is a list of
# VarName that are variables used in the openmp clauses.
class VarCollector(Transformer):
    def __init__(self):
        super(VarCollector, self).__init__()

    def PYTHON_NAME(self, args):
        return [VarName(args)]

    def const_num_or_var(self, args):
        return args[0]

    def num_threads_clause(self, args):
        (_, num_threads) = args
        if isinstance(num_threads, list):
            assert len(num_threads) == 1
            return [OnlyClauseVar(num_threads[0])]
        else:
            return None

    def __default__(self, data, children, meta):
        ret = []
        for c in children:
            if isinstance(c, list) and len(c) > 0:
                if isinstance(c[0], OnlyClauseVar):
                    ret.extend(c)
        return ret


def add_enclosing_region(func_ir, blocks, openmp_node):
    if not hasattr(func_ir, "openmp_enclosing"):
        func_ir.openmp_enclosing = {}
    for b in blocks:
        if b not in func_ir.openmp_enclosing:
            func_ir.openmp_enclosing[b] = []
        func_ir.openmp_enclosing[b].append(openmp_node)


def get_enclosing_region(func_ir, cur_block):
    if not hasattr(func_ir, "openmp_enclosing"):
        func_ir.openmp_enclosing = {}
    if cur_block in func_ir.openmp_enclosing:
        return func_ir.openmp_enclosing[cur_block]
    else:
        return None


def get_var_from_enclosing(enclosing_regions, var):
    if not enclosing_regions:
        return None
    if len(enclosing_regions) == 0:
        return None
    return enclosing_regions[-1].get_var_dsa(var)


class OpenmpVisitor(Transformer):
    target_num = 0

    def __init__(self, func_ir, blocks, blk_start, blk_end, body_blocks, loc, state):
        self.func_ir = func_ir
        self.blocks = blocks
        self.blk_start = blk_start
        self.blk_end = blk_end
        self.body_blocks = body_blocks
        self.loc = loc
        self.state = state
        super(OpenmpVisitor, self).__init__()

    # --------- Non-parser functions --------------------

    def remove_explicit_from_one(self, varset, vars_in_explicit_clauses, clauses, scope, loc):
        """Go through a set of variables and see if their non-SSA form is in an explicitly
        provided data clause.  If so, remove it from the set and add a clause so that the
        SSA form gets the same data clause.
        """
        if config.DEBUG_OPENMP >= 1:
            print("remove_explicit start:", varset, vars_in_explicit_clauses)
        diff = set()
        # For each variable inthe set.
        for v in varset:
            # Get the non-SSA form.
            flat = remove_ssa(v, scope, loc)
            # Skip non-SSA introduced variables (i.e., Python vars).
            if flat == v:
                continue
            if config.DEBUG_OPENMP >= 1:
                print("remove_explicit:", v, flat, flat in vars_in_explicit_clauses)
            # If we have the non-SSA form in an explicit data clause.
            if flat in vars_in_explicit_clauses:
                # We will remove it from the set.
                diff.add(v)
                # Copy the non-SSA variables data clause.
                ccopy = copy.copy(vars_in_explicit_clauses[flat])
                # Change the name in the clause to the SSA form.
                ccopy.arg = ir.Var(scope, v, loc)
                # Add to the clause set.
                clauses.append(ccopy)
        # Remove the vars from the set that we added a clause for.
        varset.difference_update(diff)
        if config.DEBUG_OPENMP >= 1:
            print("remove_explicit end:", varset)

    def remove_explicit_from_io_vars(self, inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, loc):
        """Remove vars in explicit data clauses from the auto-determined vars.
        Then call remove_explicit_from_one to take SSA variants out of the auto-determined sets
        and to create clauses so that SSA versions get the same clause as the explicit Python non-SSA var.
        """
        inputs_to_region.difference_update(vars_in_explicit_clauses.keys())
        def_but_live_out.difference_update(vars_in_explicit_clauses.keys())
        private_to_region.difference_update(vars_in_explicit_clauses.keys())
        self.remove_explicit_from_one(inputs_to_region, vars_in_explicit_clauses, clauses, scope, loc)
        self.remove_explicit_from_one(def_but_live_out, vars_in_explicit_clauses, clauses, scope, loc)
        self.remove_explicit_from_one(private_to_region, vars_in_explicit_clauses, clauses, scope, loc)

    def find_io_vars(self, selected_blocks):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)
        # Assumes enter_with is first statement in block.
        inputs_to_region = live_map[self.blk_start]
        if config.DEBUG_OPENMP >= 1:
            print("usedefs:", usedefs)
            print("live_map:", live_map)
            print("inputs_to_region:", inputs_to_region, type(inputs_to_region))
            print("selected blocks:", selected_blocks)
        all_uses = set()
        all_defs = set()
        for label in selected_blocks:
            all_uses = all_uses.union(usedefs.usemap[label])
            all_defs = all_defs.union(usedefs.defmap[label])
        # Filter out those vars live to the region but not used within it.
        inputs_to_region = inputs_to_region.intersection(all_uses)
        def_but_live_out = all_defs.difference(inputs_to_region).intersection(live_map[self.blk_end])
        private_to_region = all_defs.difference(inputs_to_region).difference(live_map[self.blk_end])

        if config.DEBUG_OPENMP >= 1:
            print("all_uses:", all_uses)
            print("inputs_to_region:", inputs_to_region)
            print("private_to_region:", private_to_region)
            print("def_but_live_out:", def_but_live_out)
        return inputs_to_region, def_but_live_out, private_to_region

    def get_explicit_vars(self, clauses):
        ret = {}
        privates = []
        for c in clauses:
            if config.DEBUG_OPENMP >= 1:
                print("get_explicit_vars:", c, type(c))
            if isinstance(c, openmp_tag):
                if config.DEBUG_OPENMP >= 1:
                    print("arg:", c.arg, type(c.arg))
                if isinstance(c.arg, list):
                    carglist = c.arg
                else:
                    carglist = [c.arg]
                #carglist = c.arg if isinstance(c.arg, list) else [c.arg]
                for carg in carglist:
                    if config.DEBUG_OPENMP >= 1:
                        print("carg:", carg, type(carg), user_defined_var(carg), is_dsa(c.name))
                    if isinstance(carg, str) and user_defined_var(carg) and is_dsa(c.name):
                        ret[carg] = c
                        if is_private(c.name):
                            privates.append(carg)
        return ret, privates

    def filter_unused_vars(self, clauses, used_vars):
        new_clauses = []
        for c in clauses:
            if config.DEBUG_OPENMP >= 1:
                print("filter_unused_vars:", c, type(c))
            if isinstance(c, openmp_tag):
                if config.DEBUG_OPENMP >= 1:
                    print("arg:", c.arg, type(c.arg))
                assert not isinstance(c.arg, list)
                if config.DEBUG_OPENMP >= 1:
                    print("c.arg:", c.arg, type(c.arg), user_defined_var(c.arg), is_dsa(c.name))

                if isinstance(c.arg, str) and user_defined_var(c.arg) and is_dsa(c.name):
                    if c.arg in used_vars:
                        new_clauses.append(c)
                else:
                    new_clauses.append(c)
        return new_clauses

    def get_clause_privates(self, clauses, def_but_live_out, scope, loc):
        # Get all the private clauses from the whole set of clauses.
        private_clauses_vars = [remove_privatized(x.arg) for x in clauses if x.name in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE"]]
        #private_clauses_vars = [remove_privatized(x.arg) for x in clauses if x.name in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.LASTPRIVATE"]]
        ret = {}
        # Get a mapping of vars in private clauses to the SSA version of variable exiting the region.
        for lo in def_but_live_out:
            without_ssa = remove_ssa(lo, scope, loc)
            if without_ssa in private_clauses_vars:
                ret[without_ssa] = lo
        return ret

    def make_implicit_explicit(self, scope, vars_in_explicit, explicit_clauses, gen_shared, inputs_to_region, def_but_live_out, private_to_region, for_task=False):
        #unversioned_privates = set() # we get rid of SSA on the first openmp region so no SSA forms should be here
        if gen_shared:
            for var_name in inputs_to_region:
                if for_task != False and get_var_from_enclosing(for_task, var_name) != "QUAL.OMP.SHARED":
                    explicit_clauses.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", var_name))
                else:
                    explicit_clauses.append(openmp_tag("QUAL.OMP.SHARED", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

            for var_name in def_but_live_out:
                if for_task != False and get_var_from_enclosing(for_task, var_name) != "QUAL.OMP.SHARED":
                    explicit_clauses.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", var_name))
                else:
                    explicit_clauses.append(openmp_tag("QUAL.OMP.SHARED", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

            # What to do below for task regions?
            for var_name in private_to_region:
                temp_var = ir.Var(scope, var_name, self.loc)
                if not is_internal_var(temp_var):
                    if config.OPENMP_SHARED_PRIVATE_REGION == 0:
                        #unver_var = temp_var.unversioned_name
                        #if unver_var not in unversioned_privates:
                        #    explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", unver_var))
                        #    vars_in_explicit[unver_var] = explicit_clauses[-1]
                        #    unversioned_privates.add(unver_var)
                        explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", var_name))
                        vars_in_explicit[var_name] = explicit_clauses[-1]
                    else:
                        explicit_clauses.append(openmp_tag("QUAL.OMP.SHARED", var_name))
                        vars_in_explicit[var_name] = explicit_clauses[-1]

        for var_name in private_to_region:
            temp_var = ir.Var(scope, var_name, self.loc)
            if is_internal_var(temp_var):
                #unver_var = temp_var.unversioned_name
                #if unver_var not in unversioned_privates:
                #    explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", unver_var))
                #    vars_in_explicit[unver_var] = explicit_clauses[-1]
                #    unversioned_privates.add(unver_var)
                explicit_clauses.append(openmp_tag("QUAL.OMP.PRIVATE", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

    def make_implicit_explicit_target(self, scope, vars_in_explicit, explicit_clauses, gen_shared, inputs_to_region, def_but_live_out, private_to_region):
        #unversioned_privates = set() # we get rid of SSA on the first openmp region so no SSA forms should be here
        if gen_shared:
            for var_name in inputs_to_region:
                explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]
            for var_name in def_but_live_out:
                explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]
            for var_name in private_to_region:
                temp_var = ir.Var(scope, var_name, self.loc)
                if not is_internal_var(temp_var):
                    if config.OPENMP_SHARED_PRIVATE_REGION == 0:
                        explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                        vars_in_explicit[var_name] = explicit_clauses[-1]
                    else:
                        explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                        vars_in_explicit[var_name] = explicit_clauses[-1]

        for var_name in private_to_region:
            temp_var = ir.Var(scope, var_name, self.loc)
            if is_internal_var(temp_var):
                explicit_clauses.append(openmp_tag("QUAL.OMP.TARGET.IMPLICIT" if user_defined_var(var_name) else "QUAL.OMP.PRIVATE", var_name))
                vars_in_explicit[var_name] = explicit_clauses[-1]

    def add_variables_to_start(self, scope, vars_in_explicit, explicit_clauses, gen_shared, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region):
        start_tags.extend(explicit_clauses)
        for var in vars_in_explicit:
            if not is_private(vars_in_explicit[var].name):
                evar = ir.Var(scope, var, self.loc)
                keep_alive.append(ir.Assign(evar, evar, self.loc))

        if gen_shared:
            for itr in inputs_to_region:
                itr_var = ir.Var(scope, itr, self.loc)
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
            for itr in def_but_live_out:
                itr_var = ir.Var(scope, itr, self.loc)
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
            for ptr in private_to_region:
                itr_var = ir.Var(scope, ptr, self.loc)
                if not is_internal_var(itr_var):
                    if config.OPENMP_SHARED_PRIVATE_REGION == 0:
                        start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", itr_var))
                    else:
                        start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                        keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
        for ptr in private_to_region:
            itr_var = ir.Var(scope, ptr, self.loc)
            if is_internal_var(itr_var):
                start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", itr_var))

    def add_explicits_to_start(self, scope, vars_in_explicit, explicit_clauses, gen_shared, start_tags, keep_alive):
        start_tags.extend(explicit_clauses)
        for var in vars_in_explicit:
            if not is_private(vars_in_explicit[var].name):
                evar = ir.Var(scope, var, self.loc)
                evar_copy = scope.redefine("evar_copy", self.loc)
                keep_alive.append(ir.Assign(evar, evar_copy, self.loc))
                #keep_alive.append(ir.Assign(evar, evar, self.loc))

    def flatten(self, all_clauses, start_block):
        if config.DEBUG_OPENMP >= 1:
            print("flatten", id(start_block))
        incoming_clauses = [remove_indirections(x) for x in all_clauses]
        clauses = []
        default_shared = True
        for clause in incoming_clauses:
            if config.DEBUG_OPENMP >= 1:
                print("clause:", clause, type(clause))
            if isinstance(clause, openmp_tag):
                clauses.append(clause)
            elif isinstance(clause, list):
                clauses.extend(remove_indirections(clause))
            elif isinstance(clause, default_shared_val):
                default_shared = clause.val
                if config.DEBUG_OPENMP >= 1:
                    print("got new default_shared:", clause.val)
            else:
                if config.DEBUG_OPENMP >= 1:
                    print("Unknown clause type in incoming_clauses", clause, type(clause))
                assert(0)

        if hasattr(start_block, "openmp_replace_vardict"):
            for clause in clauses:
                #print("flatten out clause:", clause, clause.arg, type(clause.arg))
                for vardict in start_block.openmp_replace_vardict:
                    if clause.arg in vardict:
                        #print("clause.arg in vardict:", clause.arg, type(clause.arg), vardict[clause.arg], type(vardict[clause.arg]))
                        clause.arg = vardict[clause.arg].name

        return clauses, default_shared

    def add_replacement(self, blocks, replace_vardict):
        for b in blocks.values():
            if not hasattr(b, "openmp_replace_vardict"):
                b.openmp_replace_vardict = []
            b.openmp_replace_vardict.append(replace_vardict)

    def replace_private_vars(self, blocks, all_explicits, explicit_privates, clauses, scope, loc, orig_inputs_to_region, for_target=False):
        replace_vardict = {}
        # Generate a new Numba privatized variable for each openmp private variable.
        for exp_priv in explicit_privates:
            replace_vardict[exp_priv] = ir.Var(scope, exp_priv + "%privatized", loc)
            #replace_vardict[exp_priv] = ir.Var(scope, exp_priv + "%privatized." + str(get_unique()), loc)
        # Get all the blocks in this openmp region and replace the original variable with the privatized one.
        block_dict = {k: v for k, v in self.blocks.items() if k in blocks}
        replace_ssa_vars(block_dict, replace_vardict)
        self.add_replacement(block_dict, replace_vardict)

        new_shared_clauses = []
        copying_ir = []
        copying_ir_before = []
        lastprivate_copying = []
        def do_copy(orig_name, private_name):
            g_copy_var = scope.redefine("$copy_g_var", loc)
            g_copy = ir.Global("openmp_copy", openmp_copy, loc)
            #g_copy = ir.Global("numba", numba, loc)
            g_copy_assign = ir.Assign(g_copy, g_copy_var, loc)

            """
            attr_var1 = scope.redefine("$copy_attr_var", loc)
            attr_getattr1 = ir.Expr.getattr(g_copy_var, 'openmp', loc)
            attr_assign1 = ir.Assign(attr_getattr1, attr_var1, loc)

            attr_var2 = scope.redefine("$copy_attr_var", loc)
            attr_getattr2 = ir.Expr.getattr(g_copy_var, 'openmp_copy', loc)
            attr_assign2 = ir.Assign(attr_getattr2, attr_var2, loc)
            """

            #copy_call = ir.Expr.call(attr_var2, [orig_name], (), loc)
            copy_call = ir.Expr.call(g_copy_var, [orig_name], (), loc)
            #copy_assign = ir.Assign(orig_name, private_name, loc)
            copy_assign = ir.Assign(copy_call, private_name, loc)
            return [g_copy_assign, copy_assign]
            #return [g_copy_assign, attr_assign1, attr_assign2, copy_assign]

        if config.DEBUG_OPENMP >= 1:
            print("replace_vardict:", replace_vardict)
            print("all_explicits:", all_explicits)
            print("explicit_privates:", explicit_privates)
            for c in clauses:
                print("clauses:", c)

        def handle_firstprivate(carg, new_shared_clauses, all_explicits, copying_ir, replace_vardict, clause):
            new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", carg))
            all_explicits[c.arg] = new_shared_clauses[-1]
            copying_ir.extend(do_copy(ir.Var(scope, carg, loc), replace_vardict[carg]))
            # This one doesn't really do anything except avoid a Numba decref error for arrays.
            #copying_ir_before.append(ir.Assign(ir.Var(scope, carg, loc), replace_vardict[carg], loc))

        def handle_lastprivate(carg, new_shared_clauses, all_explicits, lastprivate_copying, replace_vardict, clause):
            new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", carg))
            all_explicits[carg] = new_shared_clauses[-1]
            lastprivate_copying.extend(do_copy(replace_vardict[c.arg], ir.Var(scope, c.arg, loc)))
            # This one doesn't really do anything except avoid a Numba decref error for arrays.
            #copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))

        for c in clauses:
            if isinstance(c.arg, str) and c.arg in replace_vardict:
                if config.DEBUG_OPENMP >= 1:
                    print("c.arg str:", c.arg, type(c.arg))
                del all_explicits[c.arg]
                if for_target:
                    if c.name == "QUAL.OMP.PRIVATE":
                        # For typing.
                        if c.arg in orig_inputs_to_region:
                            copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                    elif c.name == "QUAL.OMP.FIRSTPRIVATE":
                        pass
                        """
                        new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", c.arg))
                        all_explicits[c.arg] = new_shared_clauses[-1]
                        copying_ir.extend(do_copy(ir.Var(scope, c.arg, loc), replace_vardict[c.arg]))
                        # This one doesn't really do anything except avoid a Numba decref error for arrays.
                        copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                        c.name = "QUAL.OMP.PRIVATE"
                        """
                else:
                    if c.name == "QUAL.OMP.PRIVATE":
                        # For typing.
                        if c.arg in orig_inputs_to_region:
                            copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                    elif c.name == "QUAL.OMP.FIRSTPRIVATE":
                        handle_firstprivate(c.arg, new_shared_clauses, all_explicits, copying_ir, replace_vardict, c)
                        c.name = "QUAL.OMP.PRIVATE"
                        """
                        new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", c.arg))
                        all_explicits[c.arg] = new_shared_clauses[-1]
                        copying_ir.extend(do_copy(ir.Var(scope, c.arg, loc), replace_vardict[c.arg]))
                        # This one doesn't really do anything except avoid a Numba decref error for arrays.
                        copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                        c.name = "QUAL.OMP.PRIVATE"
                        """
                    elif c.name == "QUAL.OMP.LASTPRIVATE":
                        handle_lastprivate(c.arg, new_shared_clauses, all_explicits, lastprivate_copying, replace_vardict, c)
                        c.name = "QUAL.OMP.PRIVATE"
                        """
                        new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", c.arg))
                        all_explicits[c.arg] = new_shared_clauses[-1]
                        lastprivate_copying.extend(do_copy(replace_vardict[c.arg], ir.Var(scope, c.arg, loc)))
                        # This one doesn't really do anything except avoid a Numba decref error for arrays.
                        #copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                        c.name = "QUAL.OMP.PRIVATE"
                        """
                c.arg = replace_vardict[c.arg].name
                all_explicits[c.arg] = c
            elif isinstance(c.arg, list):
                for i in range(len(c.arg)):
                    carg = c.arg[i]
                    if isinstance(carg, str) and carg in replace_vardict:
                        # If there is a list and some vars are replace and others
                        # not then we need to split the list here so that the
                        # ones that are private can change their clause name
                        # to private.
                        assert(False)
                        if config.DEBUG_OPENMP >= 1:
                            print("c.arg list of str:", c.arg, type(c.arg))
                        del all_explicits[carg]
                        if c.name == "QUAL.OMP.FIRSTPRIVATE":
                            handle_firstprivate(carg, new_shared_clauses, all_explicits, copying_ir, replace_vardict, c)
                            """
                            new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", carg))
                            all_explicits[carg] = new_shared_clauses[-1]
                            copying_ir.extend(do_copy(ir.Var(scope, carg, loc), replace_vardict[carg]))
                            # This one doesn't really do anything except avoid a Numba decref error for arrays.
                            copying_ir_before.append(ir.Assign(ir.Var(scope, carg, loc), replace_vardict[carg], loc))
                            """
                        elif c.name == "QUAL.OMP.LASTPRIVATE":
                            handle_lastprivate(carg, new_shared_clauses, all_explicits, lastprivate_copying, replace_vardict, c)
                            """
                            new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", carg))
                            all_explicits[carg] = new_shared_clauses[-1]
                            copying_ir.extend(do_copy(ir.Var(scope, carg, loc), replace_vardict[carg]))
                            # This one doesn't really do anything except avoid a Numba decref error for arrays.
                            copying_ir_before.append(ir.Assign(ir.Var(scope, carg, loc), replace_vardict[carg], loc))
                            """
                        newcarg = replace_vardict[carg].name
                        c.arg[i] = newcarg
                        all_explicits[newcarg] = c

        if config.DEBUG_OPENMP >= 1:
            for c in clauses:
                print("clauses:", c)
            for c in new_shared_clauses:
                print("new_shared_clauses:", c)

        clauses.extend(new_shared_clauses)
        if config.DEBUG_OPENMP >= 1:
            for c in clauses:
                print("clauses:", c)
        return replace_vardict, copying_ir, copying_ir_before, lastprivate_copying

    def prepare_for_directive(self, clauses, vars_in_explicit_clauses, before_start, after_start, start_tags, end_tags, scope):
        call_table, _ = get_call_table(self.blocks)
        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)

        all_loops = cfg.loops()
        if config.DEBUG_OPENMP >= 1:
            print("all_loops:", all_loops)
            print("live_map:", live_map)
        loops = {}
        # Find the outer-most loop in this OpenMP region.
        for k, v in all_loops.items():
            if v.header >= self.blk_start and v.header <= self.blk_end:
                loops[k] = v
        loops = list(find_top_level_loops(cfg, loops=loops))

        if config.DEBUG_OPENMP >= 1:
            print("loops:", loops)
        if len(loops) != 1:
            raise ParallelForWrongLoopCount(f"OpenMP parallel for regions must contain exactly one range based loop.  The parallel for at line {self.loc} contains {len(loops)} loops.")

        def _get_loop_kind(func_var, call_table):
            if func_var not in call_table:
                return False
            call = call_table[func_var]
            if len(call) == 0:
                return False

            return call[0] # or call[0] == prange
                    #or call[0] == 'internal_prange' or call[0] == internal_prange
                    #$or call[0] == 'pndindex' or call[0] == pndindex)

        loop = loops[0]
        entry = list(loop.entries)[0]
        header = loop.header
        exit = list(loop.exits)[0]

        loop_blocks_for_io = loop.entries.union(loop.body)
        loop_blocks_for_io_minus_entry = loop_blocks_for_io - {entry}
        non_loop_blocks = set(self.body_blocks)
        non_loop_blocks.difference_update(loop_blocks_for_io)
        #non_loop_blocks.difference_update({exit})

        if config.DEBUG_OPENMP >= 1:
            print("non_loop_blocks:", non_loop_blocks, "entry:", entry, self.body_blocks)

        first_stmt = self.blocks[entry].body[0]
        if not isinstance(first_stmt, ir.Assign) or not isinstance(first_stmt.value, ir.Global) or first_stmt.value.name != "range":
            raise ParallelForExtraCode(f"Extra code near line {self.loc} is not allowed before or after the loop in an OpenMP parallel for region.")

        live_end = live_map[self.blk_end]
        for non_loop_block in non_loop_blocks:
            nlb = self.blocks[non_loop_block]
            if isinstance(nlb.body[0], ir.Jump):
                # Non-loop empty blocks are fine.
                continue
            if isinstance(nlb.body[-1], ir.Jump) and nlb.body[-1].target == self.blk_end:
                # Loop through all statements in block that jumps to the end of the region.
                # If those are all assignments where the LHS is dead then they are safe.
                for nlb_stmt in nlb.body[:-1]:
                    if not isinstance(nlb_stmt, ir.Assign):
                        break  # Non-assignment is not known to be safe...will fallthrough to raise exception.
                    if nlb_stmt.target.name in live_end:
                        break  # Non-dead variables in assignment is not safe...will fallthrough to raise exception.
                else:
                    continue
            raise ParallelForExtraCode(f"Extra code near line {self.loc} is not allowed before or after the loop in an OpenMP parallel for region.")

        if config.DEBUG_OPENMP >= 1:
            print("loop_blocks_for_io:", loop_blocks_for_io, entry, exit)
            print("non_loop_blocks:", non_loop_blocks)

        entry_block = self.blocks[entry]
        exit_block = self.blocks[exit]
        header_block = self.blocks[header]

        latch_block_num = max(self.blocks.keys()) + 1

        # We have to reformat the Numba style of loop to the only form that xmain openmp supports.
        header_preds = [x[0] for x in cfg.predecessors(header)]
        entry_preds = list(set(header_preds).difference(loop.body))
        back_blocks = list(set(header_preds).intersection(loop.body))
        if config.DEBUG_OPENMP >= 1:
            print("header_preds:", header_preds)
            print("entry_preds:", entry_preds)
            print("back_blocks:", back_blocks)
        assert(len(entry_preds) == 1)
        entry_pred_label = entry_preds[0]
        entry_pred = self.blocks[entry_pred_label]
        header_branch = header_block.body[-1]
        post_header = {header_branch.truebr, header_branch.falsebr}
        post_header.remove(exit)
        if config.DEBUG_OPENMP >= 1:
            print("post_header:", post_header)
        post_header = self.blocks[list(post_header)[0]]
        if config.DEBUG_OPENMP >= 1:
            print("post_header:", post_header)

        normalized = True

        for inst_num, inst in enumerate(entry_block.body):
            if (isinstance(inst, ir.Assign)
                    and isinstance(inst.value, ir.Expr)
                    and inst.value.op == 'call'):
                loop_kind = _get_loop_kind(inst.value.func.name, call_table)
                if config.DEBUG_OPENMP >= 1:
                    print("loop_kind:", loop_kind)
                if loop_kind != False and loop_kind == range:
                    range_inst = inst
                    range_args = inst.value.args
                    if config.DEBUG_OPENMP >= 1:
                        print("found one", loop_kind, inst, range_args)

                    #----------------------------------------------
                    # Find getiter instruction for this range.
                    for entry_inst in entry_block.body[inst_num+1:]:
                        if (isinstance(entry_inst, ir.Assign) and
                            isinstance(entry_inst.value, ir.Expr) and
                            entry_inst.value.op == 'getiter' and
                            entry_inst.value.value == range_inst.target):
                            getiter_inst = entry_inst
                            break
                    assert(getiter_inst)
                    if config.DEBUG_OPENMP >= 1:
                        print("getiter_inst:", getiter_inst)
                    #----------------------------------------------
                    assert(len(header_block.body) > 3)
                    if config.DEBUG_OPENMP >= 1:
                        print("header block before removing Numba range vars:")
                        dump_block(header, header_block)

                    for ii in range(len(header_block.body)):
                        ii_inst = header_block.body[ii]
                        if (isinstance(ii_inst, ir.Assign) and
                            isinstance(ii_inst.value, ir.Expr) and
                            ii_inst.value.op == 'iternext'):
                            iter_num = ii
                            break

                    iternext_inst = header_block.body[iter_num]
                    pair_first_inst = header_block.body[iter_num + 1]
                    pair_second_inst = header_block.body[iter_num + 2]

                    assert(isinstance(iternext_inst, ir.Assign) and isinstance(iternext_inst.value, ir.Expr) and iternext_inst.value.op == 'iternext')
                    assert(isinstance(pair_first_inst, ir.Assign) and isinstance(pair_first_inst.value, ir.Expr) and pair_first_inst.value.op == 'pair_first')
                    assert(isinstance(pair_second_inst, ir.Assign) and isinstance(pair_second_inst.value, ir.Expr) and pair_second_inst.value.op == 'pair_second')
                    # Remove those nodes from the IR.
                    header_block.body = header_block.body[:iter_num] + header_block.body[iter_num+3:]
                    if config.DEBUG_OPENMP >= 1:
                        print("header block after removing Numba range vars:")
                        dump_block(header, header_block)

                    loop_index = pair_first_inst.target
                    if config.DEBUG_OPENMP >= 1:
                        print("loop_index:", loop_index, type(loop_index))
                    # The loop_index from Numba's perspective is not what it is from the
                    # programmer's perspective.  The OpenMP loop index is always private so
                    # we need to start from Numba's loop index (e.g., $48for_iter.3) and
                    # trace assignments from that through the header block and then find
                    # the first such assignment in the first loop block that the header
                    # branches to.
                    latest_index = loop_index
                    for hinst in header_block.body:
                        if isinstance(hinst, ir.Assign) and isinstance(hinst.value, ir.Var):
                            if hinst.value.name == latest_index.name:
                                latest_index = hinst.target
                    for phinst in post_header.body:
                        if isinstance(phinst, ir.Assign) and isinstance(phinst.value, ir.Var):
                            if phinst.value.name == latest_index.name:
                                latest_index = phinst.target
                                break
                    if config.DEBUG_OPENMP >= 1:
                        print("latest_index:", latest_index, type(latest_index))

                    if latest_index.name not in vars_in_explicit_clauses:
                        new_index_clause = openmp_tag("QUAL.OMP.PRIVATE", ir.Var(loop_index.scope, latest_index.name, inst.loc))
                        clauses.append(new_index_clause)
                        vars_in_explicit_clauses[latest_index.name] = new_index_clause
                    else:
                        if vars_in_explicit_clauses[latest_index.name].name != "QUAL.OMP.PRIVATE":
                            pass
                            # throw error?  FIX ME

                    if config.DEBUG_OPENMP >= 1:
                        for clause in clauses:
                            print("post-latest_index clauses:", clause)

                    start = 0
                    step = 1
                    size_var = range_args[0]
                    if len(range_args) == 2:
                        start = range_args[0]
                        size_var = range_args[1]
                    if len(range_args) == 3:
                        start = range_args[0]
                        size_var = range_args[1]
                        try:
                            step = self.func_ir.get_definition(range_args[2])
                        except KeyError:
                            raise NotImplementedError(
                                "Only known step size is supported for prange")
                        if not isinstance(step, ir.Const):
                            raise NotImplementedError(
                                "Only constant step size is supported for prange")
                        step = step.value
#                        if step != 1:
#                            print("unsupported step:", step, type(step))
#                            raise NotImplementedError(
#                                "Only constant step size of 1 is supported for prange")

                    #assert(start == 0 or (isinstance(start, ir.Const) and start.value == 0))
                    if config.DEBUG_OPENMP >= 1:
                        print("size_var:", size_var, type(size_var))

                    omp_lb_var = loop_index.scope.redefine("$omp_lb", inst.loc)
                    before_start.append(ir.Assign(ir.Const(0, inst.loc), omp_lb_var, inst.loc))

                    omp_iv_var = loop_index.scope.redefine("$omp_iv", inst.loc)
                    #before_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))
                    after_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))

                    types_mod_var = loop_index.scope.redefine("$numba_types_mod", inst.loc)
                    types_mod = ir.Global('types', types, inst.loc)
                    types_mod_assign = ir.Assign(types_mod, types_mod_var, inst.loc)
                    before_start.append(types_mod_assign)

                    int64_var = loop_index.scope.redefine("$int64_var", inst.loc)
                    int64_getattr = ir.Expr.getattr(types_mod_var, 'int64', inst.loc)
                    int64_assign = ir.Assign(int64_getattr, int64_var, inst.loc)
                    before_start.append(int64_assign)

                    itercount_var = loop_index.scope.redefine("$itercount", inst.loc)
                    itercount_expr = ir.Expr.itercount(getiter_inst.target, inst.loc)
                    before_start.append(ir.Assign(itercount_expr, itercount_var, inst.loc))

                    omp_ub_var = loop_index.scope.redefine("$omp_ub", inst.loc)
                    omp_ub_expr = ir.Expr.call(int64_var, [itercount_var], (), inst.loc)
                    before_start.append(ir.Assign(omp_ub_expr, omp_ub_var, inst.loc))

                    const1_var = loop_index.scope.redefine("$const1", inst.loc)
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", const1_var))
                    const1_assign = ir.Assign(ir.Const(1, inst.loc), const1_var, inst.loc)
                    before_start.append(const1_assign)
                    count_add_1 = ir.Expr.binop(operator.sub, omp_ub_var, const1_var, inst.loc)
                    before_start.append(ir.Assign(count_add_1, omp_ub_var, inst.loc))

#                    before_start.append(ir.Print([omp_ub_var], None, inst.loc))

                    omp_start_var = loop_index.scope.redefine("$omp_start", inst.loc)
                    if start == 0:
                        start = ir.Const(start, inst.loc)
                    before_start.append(ir.Assign(start, omp_start_var, inst.loc))

                    # ---------- Create latch block -------------------------------
                    latch_iv = omp_iv_var

                    latch_block = ir.Block(scope, inst.loc)
                    const1_var = loop_index.scope.redefine("$const1", inst.loc)
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", const1_var))
                    const1_assign = ir.Assign(ir.Const(1, inst.loc), const1_var, inst.loc)
                    latch_block.body.append(const1_assign)
                    latch_assign = ir.Assign(
                        ir.Expr.binop(
                            operator.add,
                            omp_iv_var,
                            const1_var,
                            inst.loc
                        ),
                        latch_iv,
                        inst.loc
                    )
                    latch_block.body.append(latch_assign)
                    latch_block.body.append(ir.Jump(header, inst.loc))

                    self.blocks[latch_block_num] = latch_block
                    for bb in back_blocks:
                        if False:
                            str_var = scope.redefine("$str_var", inst.loc)
                            str_const = ir.Const("mid start:", inst.loc)
                            str_assign = ir.Assign(str_const, str_var, inst.loc)
                            str_print = ir.Print([str_var, size_var], None, inst.loc)
                            #before_start.append(str_assign)
                            #before_start.append(str_print)
                            self.blocks[bb].body = self.blocks[bb].body[:-1] + [str_assign, str_print, ir.Jump(latch_block_num, inst.loc)]
                        else:
                            self.blocks[bb].body[-1] = ir.Jump(latch_block_num, inst.loc)
                    # -------------------------------------------------------------

                    # ---------- Header Manipulation ------------------------------
                    step_var = loop_index.scope.redefine("$step_var", inst.loc)
                    detect_step_assign = ir.Assign(ir.Const(0, inst.loc), step_var, inst.loc)
                    after_start.append(detect_step_assign)

                    step_assign = ir.Assign(ir.Const(step, inst.loc), step_var, inst.loc)
                    scale_var = loop_index.scope.redefine("$scale", inst.loc)
                    fake_iternext = ir.Assign(ir.Const(0, inst.loc), iternext_inst.target, inst.loc)
                    fake_second = ir.Assign(ir.Const(0, inst.loc), pair_second_inst.target, inst.loc)
                    scale_assign = ir.Assign(ir.Expr.binop(operator.mul, step_var, omp_iv_var, inst.loc), scale_var, inst.loc)
                    unnormalize_iv = ir.Assign(ir.Expr.binop(operator.add, omp_start_var, scale_var, inst.loc), loop_index, inst.loc)
                    cmp_var = loop_index.scope.redefine("$cmp", inst.loc)
                    iv_lte_ub = ir.Assign(ir.Expr.binop(operator.le, omp_iv_var, omp_ub_var, inst.loc), cmp_var, inst.loc)
                    old_branch = header_block.body[-1]
                    new_branch = ir.Branch(cmp_var, old_branch.truebr, old_branch.falsebr, old_branch.loc)
                    body_label = old_branch.truebr
                    first_body_block = self.blocks[body_label]
                    new_end = [iv_lte_ub, new_branch]
                    # Turn this on to add printing to help debug at runtime.
                    if False:
                        str_var = loop_index.scope.redefine("$str_var", inst.loc)
                        str_const = ir.Const("header1:", inst.loc)
                        str_assign = ir.Assign(str_const, str_var, inst.loc)
                        new_end.append(str_assign)
                        str_print = ir.Print([str_var, omp_start_var, omp_iv_var], None, inst.loc)
                        new_end.append(str_print)

                    # Prepend original contents of header into the first body block minus the comparison
                    first_body_block.body = [fake_iternext, fake_second, step_assign, scale_assign, unnormalize_iv] + header_block.body[:-1] + first_body_block.body

                    header_block.body = new_end
                    #header_block.body = [fake_iternext, fake_second, unnormalize_iv] + header_block.body[:-1] + new_end

                    # -------------------------------------------------------------

                    #const_start_var = loop_index.scope.redefine("$const_start", inst.loc)
                    #before_start.append(ir.Assign(ir.Const(0, inst.loc), const_start_var, inst.loc))
                    #start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", const_start_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", omp_iv_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", omp_ub_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_lb_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_start_var.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", loop_index.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", size_var.name))
                    return True, loop_blocks_for_io, loop_blocks_for_io_minus_entry, entry_pred, exit_block, inst, size_var, step_var, latest_index, loop_index

        return False, None, None, None, None, None, None, None, None, None

    def some_for_directive(self, args, main_start_tag, main_end_tag, first_clause, gen_shared):
        sblk = self.blocks[self.blk_start]
        scope = sblk.scope
        eblk = self.blocks[self.blk_end]

        #clauses = []
        #default_shared = True
        if config.DEBUG_OPENMP >= 1:
            print("some_for_directive", self.body_blocks)
        clauses, default_shared = self.flatten(args[first_clause:], sblk)

        if config.DEBUG_OPENMP >= 1:
            print("visit", main_start_tag, args, type(args), default_shared)
            for clause in clauses:
                print("post-process clauses:", clause)

        if len(list(filter(lambda x: x.name == "QUAL.OMP.NUM_THREADS", clauses))) > 1:
            raise MultipleNumThreadsClauses(f"Multiple num_threads clauses near line {self.loc} is not allowed in an OpenMP parallel region.")

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses), explicit_privates)

        before_start = []
        after_start = []
        start_tags = [ openmp_tag(main_start_tag) ]
        end_tags   = [ openmp_tag(main_end_tag) ]

        if config.DEBUG_OPENMP >= 1:
            print("pre-replace vars_in_explicit_clauses:", vars_in_explicit_clauses)
            print("pre-replace explicit_privates:", explicit_privates)
            for c in clauses:
                print("pre-replace clauses:", c)

        prepare_out = self.prepare_for_directive(clauses,
                                                 vars_in_explicit_clauses,
                                                 before_start,
                                                 after_start,
                                                 start_tags,
                                                 end_tags,
                                                 scope)
        found_loop, loop_blocks_for_io, loop_blocks_for_io_minus_entry, entry_pred, exit_block, inst, size_var, step_var, latest_index, loop_index = prepare_out

        assert(found_loop)

        # ----------- DSA handling ----------------------

        keep_alive = []
        # Do an analysis to get variable use information coming into and out of the region.
        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(loop_blocks_for_io)
        #inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(loop_blocks_for_io_minus_entry)
        if config.DEBUG_OPENMP >= 1:
            print("initial find_io_vars:", inputs_to_region, def_but_live_out, private_to_region)
        orig_inputs_to_region = copy.copy(inputs_to_region)
        live_out_copy = copy.copy(def_but_live_out)

        priv_saves = []
        priv_restores = []
        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, live_out_copy, scope, self.loc)
        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = scope.redefine("$"+cp, self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)
        if config.DEBUG_OPENMP >= 1:
            print("post remove explicit:", inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses)
            for c in clauses:
                print("post remove explicit clauses:", c)

        if not default_shared and (
            has_user_defined_var(inputs_to_region) or
            has_user_defined_var(def_but_live_out) or
            has_user_defined_var(private_to_region)):
            user_defined_inputs = get_user_defined_var(inputs_to_region)
            user_defined_def_live = get_user_defined_var(def_but_live_out)
            user_defined_private = get_user_defined_var(private_to_region)
            if config.DEBUG_OPENMP >= 1:
                print("inputs users:", user_defined_inputs)
                print("def users:", user_defined_def_live)
                print("private users:", user_defined_private)
            raise UnspecifiedVarInDefaultNone("Variables with no data env clause in OpenMP region: " + str(user_defined_inputs.union(user_defined_def_live).union(user_defined_private)))

        self.make_implicit_explicit(scope, vars_in_explicit_clauses, clauses, gen_shared, inputs_to_region, def_but_live_out, private_to_region)
        #self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, gen_shared, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

        replace_vardict, copying_ir, copying_ir_before, lastprivate_copying = self.replace_private_vars(loop_blocks_for_io_minus_entry, vars_in_explicit_clauses, explicit_privates, clauses, scope, self.loc, orig_inputs_to_region)
        before_start.extend(copying_ir_before)
        after_start.extend(copying_ir)

        if config.DEBUG_OPENMP >= 1:
            print("post-replace vars_in_explicit_clauses:", vars_in_explicit_clauses)
            print("post-replace explicit_privates:", explicit_privates)
            for c in clauses:
                print("post-replace clauses:", c)
            print("lastprivate_copying:", lastprivate_copying)
            for c in lastprivate_copying:
                print(c)
            print("copying_ir:", copying_ir)
            for c in copying_ir:
                print(c)
            print("copying_ir_before:", copying_ir_before)
            for c in copying_ir_before:
                print(c)

        self.add_explicits_to_start(scope, vars_in_explicit_clauses, clauses, gen_shared, start_tags, keep_alive)

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)

        """
        if len(lastprivate_copying) > 0:
            size_var_copy = scope.redefine("size_var_copy", inst.loc)
            before_start.append(ir.Assign(size_var, size_var_copy, inst.loc))
            if True:
                str_var = scope.redefine("$str_var", inst.loc)
                str_const = ir.Const("before start:", inst.loc)
                str_assign = ir.Assign(str_const, str_var, inst.loc)
                str_print = ir.Print([str_var, size_var_copy], None, inst.loc)
                before_start.append(str_assign)
                before_start.append(str_print)
        """

        #new_header_block.body = [or_start] + before_start + new_header_block.body[:]
        #entry_pred.body = entry_pred.body[:-1] + priv_saves + before_start + [or_start] + after_start + [entry_pred.body[-1]]
        #entry_block.body = [or_start] + before_start + entry_block.body[:]
        #entry_block.body = entry_block.body[:inst_num] + before_start + [or_start] + entry_block.body[inst_num:]
        #exit_block.body = [or_end] + priv_restores + exit_block.body
        if len(lastprivate_copying) > 0:
            new_exit_block = ir.Block(scope, inst.loc)
            new_exit_block_num = max(self.blocks.keys()) + 1
            self.blocks[new_exit_block_num] = new_exit_block
            evar_copy = scope.redefine("evar_copy", self.loc)
            keep_alive.append(ir.Assign(size_var, evar_copy, self.loc))
            #keep_alive.append(ir.Assign(size_var_copy, evar_copy, self.loc))
            new_exit_block.body = [or_end] + priv_restores + keep_alive + exit_block.body

            lastprivate_check_block = exit_block
            lastprivate_check_block.body = []

            lastprivate_copy_block = ir.Block(scope, inst.loc)
            lastprivate_copy_block_num = max(self.blocks.keys()) + 1
            self.blocks[lastprivate_copy_block_num] = lastprivate_copy_block

            #lastprivate_check_block.body.append(ir.Jump(lastprivate_copy_block_num, inst.loc))
            bool_var = scope.redefine("$bool_var", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Global("bool", bool, inst.loc), bool_var, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", bool_var.name))

            size_minus_step = scope.redefine("$size_minus_step", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.sub, size_var, step_var, inst.loc), size_minus_step, inst.loc))
            #lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.sub, size_var_copy, step_var, inst.loc), size_minus_step, inst.loc))
            #or_start.add_tag(openmp_tag("QUAL.OMP.FIRSTPRIVATE", size_var_copy.name))
            or_start.add_tag(openmp_tag("QUAL.OMP.SHARED", size_var.name))
            #or_start.add_tag(openmp_tag("QUAL.OMP.SHARED", size_var_copy.name))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", size_minus_step.name))

            cmp_var = scope.redefine("$lastiter_cmp_var", inst.loc)
            if latest_index.name in replace_vardict:
                li_privatized = replace_vardict[latest_index.name]
                lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.ge, li_privatized, size_minus_step, inst.loc), cmp_var, inst.loc))
            else:
                lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.ge, latest_index, size_minus_step, inst.loc), cmp_var, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", cmp_var.name))

            zero_var = loop_index.scope.redefine("$zero_var", inst.loc)
            zero_assign = ir.Assign(ir.Const(0, inst.loc), zero_var, inst.loc)
            lastprivate_check_block.body.append(zero_assign)
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", zero_var.name))

            did_work_var = scope.redefine("$did_work_var", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.ne, step_var, zero_var, inst.loc), did_work_var, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", did_work_var.name))

            last_iter_cmp = scope.redefine("$lastiter_cmp_var_bool", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Expr.call(bool_var, [cmp_var], (), inst.loc), last_iter_cmp, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", last_iter_cmp.name))

            did_work_cmp = scope.redefine("$did_work_var_bool", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Expr.call(bool_var, [did_work_var], (), inst.loc), did_work_cmp, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", did_work_cmp.name))

            and_var = scope.redefine("$and_var", inst.loc)
            lastprivate_check_block.body.append(ir.Assign(ir.Expr.binop(operator.and_, last_iter_cmp, did_work_cmp, inst.loc), and_var, inst.loc))
            or_start.add_tag(openmp_tag("QUAL.OMP.PRIVATE", and_var.name))

            if False:
                str_var = scope.redefine("$str_var", inst.loc)
                str_const = ir.Const("lastiter check:", inst.loc)
                str_assign = ir.Assign(str_const, str_var, inst.loc)
                lastprivate_check_block.body.append(str_assign)
                str_print = ir.Print([str_var, latest_index, size_var, last_iter_cmp, omp_lb_var, omp_ub_var, did_work_cmp, and_var, size_minus_step, step_var], None, inst.loc)
                #str_print = ir.Print([str_var, latest_index, size_var_copy, last_iter_cmp, omp_lb_var, omp_ub_var, did_work_cmp, and_var, size_minus_step, step_var], None, inst.loc)
                lastprivate_check_block.body.append(str_print)

            lastprivate_check_block.body.append(ir.Branch(and_var, lastprivate_copy_block_num, new_exit_block_num, inst.loc))

            lastprivate_copy_block.body.extend(lastprivate_copying)
            lastprivate_copy_block.body.append(ir.Jump(new_exit_block_num, inst.loc))
            entry_pred.body = entry_pred.body[:-1] + priv_saves + before_start + [or_start] + after_start + [entry_pred.body[-1]]
        else:
            entry_pred.body = entry_pred.body[:-1] + priv_saves + before_start + [or_start] + after_start + [entry_pred.body[-1]]
            exit_block.body = [or_end] + priv_restores + keep_alive + exit_block.body

        return None

    # --------- Parser functions ------------------------

    def barrier_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit barrier_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.BARRIER")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.BARRIER")], self.loc)
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def taskwait_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit taskwait_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.TASKWAIT")], 0, self.loc)
        or_start.requires_combined_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.TASKWAIT")], self.loc)
        sblk.body = [or_start] + [or_end] + sblk.body[:]

    def taskyield_directive(self, args):
        raise NotImplementedError("Taskyield currently unsupported.")

    # Don't need a rule for BARRIER.
    # Don't need a rule for TASKWAIT.
    # Don't need a rule for TASKYIELD.

    def taskgroup_directive(self, args):
        raise NotImplementedError("Taskgroup currently unsupported.")

    # Don't need a rule for taskgroup_construct.
    # Don't need a rule for TASKGROUP.

    # Don't need a rule for openmp_construct.

    #def teams_distribute_parallel_for_simd_clause(self, args):
    #    raise NotImplementedError("""Simd clause for target teams
    #                             distribute parallel loop currently unsupported.""")
    #    if config.DEBUG_OPENMP >= 1:
    #        print("visit device_clause", args, type(args))

    # Don't need a rule for for_simd_construct.

    def for_simd_directive(self, args):
        raise NotImplementedError("For simd currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit for_simd_directive", args, type(args))

    def for_simd_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit for_simd_clause",
                  args, type(args), args[0])
        return args[0]

    # Don't need a rule for parallel_for_simd_construct.

    def parallel_for_simd_directive(self, args):
        raise NotImplementedError("Parallel for simd currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_for_simd_directive", args, type(args))

    def parallel_for_simd_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_for_simd_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for target_data_construct.

    def target_data_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit target_data_directive", args, type(args))

        before_start = []
        after_start = []

        clauses, default_shared = self.flatten(args[2:], sblk)

        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        used_in_region = inputs_to_region | def_but_live_out | private_to_region
        clauses = self.filter_unused_vars(clauses, used_in_region)

        start_tags = [openmp_tag("DIR.OMP.TARGET.DATA")] + clauses
        end_tags = [openmp_tag("DIR.OMP.END.TARGET.DATA")]

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = before_start + [or_start] + after_start + sblk.body[:]
        eblk.body = [or_end] + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)

    # Don't need a rule for DATA.

    def target_data_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_data_clause", args, type(args), args[0])
        return args[0]

    def device_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit device_clause", args, type(args))
        return [openmp_tag("QUAL.OMP.DEVICE", args[0])]

    def map_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit map_clause", args, type(args), args[0])
        if args[0] in ["to", "from", "alloc", "tofrom"]:
            map_type = args[0].upper()
            var_list = args[1]
            assert(len(args) == 2)
        else:
            map_type = "TOFROM"  # is this default right?  FIX ME
            var_list = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.MAP." + map_type, var))
        return ret

    def map_type(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit map_type", args, type(args), args[0])
        return str(args[0])

    # Don't need a rule for TO.
    # Don't need a rule for FROM.
    # Don't need a rule for ALLOC.
    # Don't need a rule for TOFROM.
    # Don't need a rule for parallel_sections_construct.

    def parallel_sections_directive(self, args):
        raise NotImplementedError("Parallel sections currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_sections_directive", args, type(args))

    def parallel_sections_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for sections_construct.

    def sections_directive(self, args):
        raise NotImplementedError("Sections directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit sections_directive", args, type(args))

    # Don't need a rule for SECTIONS.

    def sections_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit sections_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for section_construct.

    def section_directive(self, args):
        raise NotImplementedError("Section directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit section_directive", args, type(args))

    # Don't need a rule for SECTION.
    # Don't need a rule for atomic_construct.

    def atomic_directive(self, args):
        raise NotImplementedError("Atomic currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit atomic_directive", args, type(args))

    # Don't need a rule for ATOMIC.

    def atomic_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit atomic_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for READ.
    # Don't need a rule for WRITE.
    # Don't need a rule for UPDATE.
    # Don't need a rule for CAPTURE.
    # Don't need a rule for seq_cst_clause.
    # Don't need a rule for critical_construct.

    def critical_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if config.DEBUG_OPENMP >= 1:
            print("visit critical_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.CRITICAL")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.CRITICAL")], self.loc)

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        inputs_to_region = {remove_ssa(x, scope, self.loc):x for x in inputs_to_region}
        def_but_live_out = {remove_ssa(x, scope, self.loc):x for x in def_but_live_out}
        common_keys = inputs_to_region.keys() & def_but_live_out.keys()
        in_def_live_out = {inputs_to_region[k]:def_but_live_out[k] for k in common_keys}
        if config.DEBUG_OPENMP >= 1:
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("in_def_live_out:", in_def_live_out)

        reset = []
        for k,v in in_def_live_out.items():
            reset.append(ir.Assign(ir.Var(scope, v, self.loc), ir.Var(scope, k, self.loc), self.loc))

        sblk.body = [or_start] + sblk.body[:]
        eblk.body = reset + [or_end] + eblk.body[:]

    # Don't need a rule for CRITICAL.
    # Don't need a rule for target_construct.
    # Don't need a rule for target_teams_distribute_parallel_for_simd_construct.

    def teams_directive(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit teams_directive", args, type(args), self.blk_start, self.blk_end)
        start_tags = [openmp_tag("DIR.OMP.TEAMS")]
        end_tags = [openmp_tag("DIR.OMP.END.TEAMS")]
        clauses = self.some_data_clause_directive(args, start_tags, end_tags, 1)

        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if config.DEBUG_OPENMP >= 1:
            print("teams enclosing_regions:", enclosing_regions)
        if enclosing_regions:
            for enclosing_region in enclosing_regions[::-1]:
                if len(self.get_clauses_by_name(enclosing_region.tags, "DIR.OMP.TARGET")) == 1:
                    nt_tag = self.get_clauses_by_name(enclosing_region.tags, "QUAL.OMP.NUM_TEAMS")
                    assert len(nt_tag) > 0
                    cur_num_team_clauses = self.get_clauses_by_name(clauses, "QUAL.OMP.NUM_TEAMS")
                    if len(cur_num_team_clauses) >= 1:
                        nt_tag[-1].arg = cur_num_team_clauses[-1].arg
                    else:
                        nt_tag[-1].arg = 0

                    nt_tag = self.get_clauses_by_name(enclosing_region.tags, "QUAL.OMP.THREAD_LIMIT")
                    assert len(nt_tag) > 0
                    cur_num_team_clauses = self.get_clauses_by_name(clauses, "QUAL.OMP.THREAD_LIMIT")
                    if len(cur_num_team_clauses) >= 1:
                        nt_tag[-1].arg = cur_num_team_clauses[-1].arg
                    else:
                        nt_tag[-1].arg = 0

                    break

        """
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if config.DEBUG_OPENMP >= 1:
            print("visit teams_directive", args, type(args))

        before_start = []
        after_start = []

        clauses, default_shared = self.flatten(args[2:], sblk)

        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))
            for v in clauses:
                print("vars_in_explicit clauses first:", v)

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        used_in_region = inputs_to_region | def_but_live_out | private_to_region
        clauses = self.filter_unused_vars(clauses, used_in_region)

        start_tags = [openmp_tag("DIR.OMP.TEAMS")] + clauses
        end_tags = [openmp_tag("DIR.OMP.END.TEAMS")]

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = before_start + [or_start] + after_start + sblk.body[:]
        eblk.body = [or_end] + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)
        """

    def target_directive(self, args):
        self.some_target_directive(args, "TARGET", 1)

    def target_teams_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS", 2)

    def target_teams_loop_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS.LOOP", 3, has_loop=True)

    def target_teams_distribute_parallel_for_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP", 5, has_loop=True)

    def target_teams_distribute_parallel_for_simd_directive(self, args):
        self.some_target_directive(args, "TARGET.TEAMS.DISTRIBUTE.PARALLEL.LOOP.SIMD", 6, has_loop=True)

    def get_clauses_by_name(self, clauses, name):
        return list(filter(lambda x: x.name == name, clauses))

    def some_target_directive(self, args, dir_tag, lexer_count, has_loop=False):
        if config.DEBUG_OPENMP >= 1:
            print("visit some_target_directive", args, type(args), self.blk_start, self.blk_end)
        target_num = OpenmpVisitor.target_num
        OpenmpVisitor.target_num += 1

        dir_start_tag = "DIR.OMP." + dir_tag
        dir_end_tag = "DIR.OMP.END." + dir_tag
        start_tags = [openmp_tag(dir_start_tag, target_num)]
        end_tags = [openmp_tag(dir_end_tag, target_num)]

        sblk = self.blocks[self.blk_start]
        clauses, _ = self.flatten(args[lexer_count:], sblk)
        if len(self.get_clauses_by_name(clauses, "QUAL.OMP.NUM_TEAMS")) == 0:
            if config.DEBUG_OPENMP >= 1:
                print("Adding NUM_TEAMS implicit clause.")
            start_tags.append(openmp_tag("QUAL.OMP.NUM_TEAMS", 1))
        if len(self.get_clauses_by_name(clauses, "QUAL.OMP.THREAD_LIMIT")) == 0:
            if config.DEBUG_OPENMP >= 1:
                print("Adding THREAD_LIMIT implicit clause.")
            start_tags.append(openmp_tag("QUAL.OMP.THREAD_LIMIT", 1))

        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("target clause:", clause)

        self.some_data_clause_directive(clauses, start_tags, end_tags, 0, has_loop=has_loop)
        #self.some_data_clause_directive(args, start_tags, end_tags, lexer_count, has_loop=has_loop)

    def some_data_clause_directive(self, args, start_tags, end_tags, lexer_count, has_loop=False):
        if config.DEBUG_OPENMP >= 1:
            print("visit some_data_clause_directive", args, type(args), self.blk_start, self.blk_end)

        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        if config.DEBUG_OPENMP >= 1:
            for clause in args[lexer_count:]:
                print("pre clause:", clause)
        clauses, _ = self.flatten(args[lexer_count:], sblk)
        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        before_start = []
        after_start = []
        for_before_start = []
        for_after_start = []

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))
            for v in clauses:
                print("vars_in_explicit clauses first:", v)

        if has_loop:
            prepare_out = self.prepare_for_directive(clauses,
                                                     vars_in_explicit_clauses,
                                                     for_before_start,
                                                     for_after_start,
                                                     start_tags,
                                                     end_tags,
                                                     scope)

            found_loop, blocks_for_io, blocks_in_region, entry_pred, exit_block, _, _, _, _, _ = prepare_out

            assert(found_loop)
        else:
            blocks_for_io = self.body_blocks
            blocks_in_region = get_blocks_between_start_end(self.blocks, self.blk_start, self.blk_end)
            entry_pred = sblk
            exit_block = eblk

        # Do an analysis to get variable use information coming into and out of the region.
        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(blocks_for_io)
        orig_inputs_to_region = copy.copy(inputs_to_region)
        live_out_copy = copy.copy(def_but_live_out)

        if config.DEBUG_OPENMP >= 1:
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("private_to_region:", private_to_region)
            for v in clauses:
                print("clause after find_io_vars:", v)

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

        if config.DEBUG_OPENMP >= 1:
            for v in clauses:
                print("clause after remove_explicit_from_io_vars:", v)

        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit before:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses before:", v)
        self.make_implicit_explicit_target(scope, vars_in_explicit_clauses, clauses, True, inputs_to_region, def_but_live_out, private_to_region)
        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit after:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses after:", v)
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("post get_explicit_vars:", explicit_privates)
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit post:", k, v)
        if config.DEBUG_OPENMP >= 1:
            print(1, "blocks_in_region:", blocks_in_region)
        replace_vardict, copying_ir, copying_ir_before, lastprivate_copying = self.replace_private_vars(blocks_in_region, vars_in_explicit_clauses, explicit_privates, clauses, scope, self.loc, orig_inputs_to_region, for_target=True)
        assert(len(lastprivate_copying) == 0)
        before_start.extend(copying_ir_before)
        after_start.extend(copying_ir)
        if config.DEBUG_OPENMP >= 1:
            for ci in copying_ir_before:
                print("copying_ir_before:", ci)
            for ci in copying_ir:
                print("copying_ir:", ci)

        priv_saves = []
        priv_restores = []
        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, live_out_copy, scope, self.loc)
        for k,v in replace_vardict.items():
            if k in orig_inputs_to_region:
                priv_saves.append(ir.Assign(ir.Var(scope, k, self.loc), v, self.loc))

        if config.DEBUG_OPENMP >= 1:
            print("replace_vardict:", replace_vardict)
            print("clause_privates:", clause_privates, type(clause_privates))
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("live_out_copy:", live_out_copy)
            print("private_to_region:", private_to_region)
            for ps in priv_saves:
                print("priv_saves:", ps)

        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = scope.redefine("$"+cp, self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        keep_alive = []
        self.add_explicits_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive)

        #or_start = openmp_region_start([openmp_tag("DIR.OMP.TARGET", target_num)] + clauses, 0, self.loc)
        #or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.TARGET", target_num)], self.loc)
        new_target_block_num = max(self.blocks.keys()) + 1

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        target_block = ir.Block(scope, self.loc)
        target_block.body = [or_start] + after_start + sblk.body[:]
        self.blocks[new_target_block_num] = target_block

        if has_loop:
            entry_pred.body = entry_pred.body[:-1] + before_start + for_before_start + [or_start] + after_start + for_after_start + [entry_pred.body[-1]]
            exit_block.body = [or_end] + priv_restores + keep_alive + exit_block.body
        else:
            sblk.body = priv_saves + before_start + [ir.Jump(new_target_block_num, self.loc)]
            eblk.body = [or_end] + priv_restores + keep_alive + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)
        return clauses

    def target_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_teams_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_distribute_parallel_for_simd_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_teams_distribute_parallel_for_simd_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_distribute_parallel_for_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_teams_distribute_parallel_for_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def target_teams_loop_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_teams_loop_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    # Don't need a rule for target_update_construct.

    def target_update_directive(self, args):
        raise NotImplementedError("Target update currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit target_update_directive", args, type(args))

    def target_update_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_update_clause", args, type(args), args[0])
        return args[0]

    def motion_clause(self, args):
        raise NotImplementedError("Motion clause currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit motion_clause", args, type(args))

    def variable_array_section_list(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit variable_array_section_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def array_section(self, args):
        raise NotImplementedError("No implementation for array sections.")
        if config.DEBUG_OPENMP >= 1:
            print("visit array_section", args, type(args))

    def array_section_subscript(self, args):
        raise NotImplementedError("No implementation for array section subscript.")
        if config.DEBUG_OPENMP >= 1:
            print("visit array_section_subscript", args, type(args))

    # Don't need a rule for TARGET.
    # Don't need a rule for single_construct.

    def single_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit single_directive", args, type(args))
        or_start = openmp_region_start([openmp_tag("DIR.OMP.SINGLE")], 0, self.loc)
        or_start.requires_acquire_release()
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.SINGLE")], self.loc)
        sblk.body = [or_start] + sblk.body[:]
        eblk.body = [or_end]   + eblk.body[:]

    def single_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit single_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for unique_single_clause.
    #def NOWAIT(self, args):
    #    return "nowait"
    # Don't need a rule for NOWAIT.
    # Don't need a rule for master_construct.

    def master_directive(self, args):
        raise NotImplementedError("Master directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit master_directive", args, type(args))

    # Don't need a rule for simd_construct.

    def simd_directive(self, args):
        raise NotImplementedError("Simd directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit simd_directive", args, type(args))

    # Don't need a rule for SIMD.

    def simd_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit simd_clause", args, type(args), args[0])
        return args[0]

    def aligned_clause(self, args):
        raise NotImplementedError("Aligned clause currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit aligned_clause", args, type(args))

    # Don't need a rule for declare_simd_construct.

    def declare_simd_directive_seq(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit declare_simd_directive_seq", args, type(args), args[0])
        return args[0]

    def declare_simd_directive(self, args):
        raise NotImplementedError("Declare simd directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit declare_simd_directive", args, type(args))

    def declare_simd_clause(self, args):
        raise NotImplementedError("Declare simd clauses currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit declare_simd_clause", args, type(args))

    # Don't need a rule for ALIGNED.

    def inbranch_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit inbranch_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for INBRANCH.
    # Don't need a rule for NOTINBRANCH.

    def uniform_clause(self, args):
        raise NotImplementedError("Uniform clause currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit uniform_clause", args, type(args))

    # Don't need a rule for UNIFORM.

    def collapse_clause(self, args):
        raise NotImplementedError("Collapse currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit collapse_clause", args, type(args))

    # Don't need a rule for COLLAPSE.
    # Don't need a rule for task_construct.
    # Don't need a rule for TASK.

    def task_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope
        clauses, _ = self.flatten(args[1:], sblk)

        before_start = []
        after_start = []

        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if config.DEBUG_OPENMP >= 1:
            print("enclosing_regions:", enclosing_regions)

        start_tags = [openmp_tag("DIR.OMP.TASK")] + clauses
        end_tags   = [openmp_tag("DIR.OMP.END.TASK")]
        keep_alive = []

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        used_in_region = inputs_to_region | def_but_live_out | private_to_region
        clauses = self.filter_unused_vars(clauses, used_in_region)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))
            for v in clauses:
                print("vars_in_explicit clauses first:", v)

        orig_inputs_to_region = copy.copy(inputs_to_region)
        live_out_copy = copy.copy(def_but_live_out)

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit before:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses before:", v)
        self.make_implicit_explicit(scope, vars_in_explicit_clauses, clauses, True, inputs_to_region, def_but_live_out, private_to_region, for_task=enclosing_regions)
        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit after:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses after:", v)
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("post get_explicit_vars:", explicit_privates)
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit post:", k, v)
        blocks_in_region = get_blocks_between_start_end(self.blocks, self.blk_start, self.blk_end)
        if config.DEBUG_OPENMP >= 1:
            print(1, "blocks_in_region:", blocks_in_region)
        #replace_vardict, copying_ir, copying_ir_before, lastprivate_copying = self.replace_private_vars(blocks_in_region, vars_in_explicit_clauses, explicit_privates, clauses, scope, self.loc, orig_inputs_to_region)
        if config.DEBUG_OPENMP >= 1:
            for v in clauses:
                print("clause after remove_explicit_from_io_vars:", v)

        #before_start.extend(copying_ir_before)
        #after_start.extend(copying_ir)

        priv_saves = []
        priv_restores = []
        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, live_out_copy, scope, self.loc)
        if config.DEBUG_OPENMP >= 1:
            print("clause_privates:", clause_privates, type(clause_privates))
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("live_out_copy:", live_out_copy)
            print("private_to_region:", private_to_region)

        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = scope.redefine("$"+cp, self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        """
        for itr in inputs_to_region:
            enclosing_dsa = get_var_from_enclosing(enclosing_regions, itr)
            if config.DEBUG_OPENMP >= 1:
                print("input_to_region:", itr, enclosing_dsa)
            if enclosing_dsa == "QUAL.OMP.SHARED":
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", ir.Var(scope, itr, self.loc)))
            else:
                start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", ir.Var(scope, itr, self.loc)))
        for itr in def_but_live_out:
            enclosing_dsa = get_var_from_enclosing(enclosing_regions, itr)
            if config.DEBUG_OPENMP >= 1:
                print("def_but_live_out:", itr, enclosing_dsa)
            itr_var = ir.Var(scope, itr, self.loc)
            if enclosing_dsa == "QUAL.OMP.SHARED":
                start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
                keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
            else:
                start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", itr_var))

        for ptr in private_to_region:
            if config.DEBUG_OPENMP >= 1:
                print("private_to_region:", ptr)
            start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", ir.Var(scope, ptr, self.loc)))
        """

        if config.DEBUG_OPENMP >= 1:
            print("visit task_directive", args, type(args), clauses)

        self.add_explicits_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive)
        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = priv_saves + before_start + [or_start] + after_start + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        eblk.body = [or_end] + priv_restores + keep_alive + eblk.body[:]

    def task_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit task_clause", args, type(args), args[0])
        return args[0]

    def unique_task_clause(self, args):
        raise NotImplementedError("Task-related clauses currently unsupported.")
        if config.DEBUG_OPENMP >= 1:
            print("visit unique_task_clause", args, type(args))

    # Don't need a rule for DEPEND.
    # Don't need a rule for FINAL.
    # Don't need a rule for UNTIED.
    # Don't need a rule for MERGEABLE.

    def dependence_type(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit dependence_type", args, type(args), args[0])
        return args[0]

    # Don't need a rule for IN.
    # Don't need a rule for OUT.
    # Don't need a rule for INOUT.

    def data_default_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_default_clause", args, type(args), args[0])
        return args[0]

    def data_sharing_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_sharing_clause", args, type(args), args[0])
        return args[0]

    def data_privatization_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_privatization_clause", args, type(args), args[0])
            if isinstance(args[0], list):
                print(args[0][0])
        return args[0]

    def data_privatization_in_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_privatization_in_clause", args, type(args), args[0])
        return args[0]

    def data_privatization_out_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_privatization_out_clause", args, type(args), args[0])
        return args[0]

    def data_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_clause", args, type(args), args[0])
        return args[0]

    def private_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit private_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.PRIVATE", var))
        return ret

    # Don't need a rule for PRIVATE.

    def copyprivate_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit copyprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYPRIVATE", var))
        return ret

    # Don't need a rule for COPYPRIVATE.

    def firstprivate_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit firstprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", var))
        return ret

    # Don't need a rule for FIRSTPRIVATE.

    def lastprivate_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit lastprivate_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.LASTPRIVATE", var))
        return ret

    # Don't need a rule for LASTPRIVATE.

    def shared_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit shared_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.SHARED", var))
        return ret

    # Don't need a rule for SHARED.

    def copyin_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit copyin_clause", args, type(args), args[0])
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.COPYIN", var))
        return ret

    # Don't need a rule for COPYIN.
    # Don't need a rule for REDUCTION.

    def data_reduction_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit data_reduction_clause", args, type(args), args[0])

        (_, red_op, red_list) = args
        ret = []
        for shared in red_list:
            ret.append(openmp_tag("QUAL.OMP.REDUCTION." + red_op, shared))
        return ret

    def default_shared_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit default_shared_clause", args, type(args))
        return default_shared_val(True)

    def default_none_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit default_none", args, type(args))
        return default_shared_val(False)

    def const_num_or_var(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit const_num_or_var", args, type(args))
        return args[0]

    # Don't need a rule for parallel_construct.

    def parallel_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        before_start = []
        after_start = []
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_directive", args, type(args))
        clauses, default_shared = self.flatten(args[1:], sblk)

        if len(list(filter(lambda x: x.name == "QUAL.OMP.NUM_THREADS", clauses))) > 1:
            raise MultipleNumThreadsClauses(f"Multiple num_threads clauses near line {self.loc} is not allowed in an OpenMP parallel region.")

        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        # ---- Back propagate THREAD_LIMIT to enclosed target region. ----
        enclosing_regions = get_enclosing_region(self.func_ir, self.blk_start)
        if config.DEBUG_OPENMP >= 1:
            print("parallel enclosing_regions:", enclosing_regions)
        if enclosing_regions:
            for enclosing_region in enclosing_regions[::-1]:
                if len(self.get_clauses_by_name(enclosing_region.tags, "DIR.OMP.TEAMS")) == 1:
                    break
                if len(self.get_clauses_by_name(enclosing_region.tags, "DIR.OMP.TARGET")) == 1:
                    nt_tag = self.get_clauses_by_name(enclosing_region.tags, "QUAL.OMP.THREAD_LIMIT")
                    assert len(nt_tag) > 0
                    cur_thread_limit_clauses = self.get_clauses_by_name(clauses, "QUAL.OMP.NUM_THREADS")
                    if len(cur_thread_limit_clauses) >= 1:
                        nt_tag[-1].arg = cur_thread_limit_clauses[-1].arg
                    else:
                        nt_tag[-1].arg = 0
                    break
        # DONE ---- Back propagate THREAD_LIMIT to enclosed target region ----

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        used_in_region = inputs_to_region | def_but_live_out | private_to_region
        clauses = self.filter_unused_vars(clauses, used_in_region)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))
            for v in clauses:
                print("vars_in_explicit clauses first:", v)

        # Do an analysis to get variable use information coming into and out of the region.
        orig_inputs_to_region = copy.copy(inputs_to_region)
        live_out_copy = copy.copy(def_but_live_out)

        if config.DEBUG_OPENMP >= 1:
            for v in clauses:
                print("clause after find_io_vars:", v)

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

        if config.DEBUG_OPENMP >= 1:
            for v in clauses:
                print("clause after remove_explicit_from_io_vars:", v)

        if not default_shared and (
            has_user_defined_var(inputs_to_region) or
            has_user_defined_var(def_but_live_out) or
            has_user_defined_var(private_to_region)):
            user_defined_inputs = get_user_defined_var(inputs_to_region)
            user_defined_def_live = get_user_defined_var(def_but_live_out)
            user_defined_private = get_user_defined_var(private_to_region)
            if config.DEBUG_OPENMP >= 1:
                print("inputs users:", user_defined_inputs)
                print("def users:", user_defined_def_live)
                print("private users:", user_defined_private)
            raise UnspecifiedVarInDefaultNone("Variables with no data env clause in OpenMP region: " + str(user_defined_inputs.union(user_defined_def_live).union(user_defined_private)))

        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit before:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses before:", v)
        self.make_implicit_explicit(scope, vars_in_explicit_clauses, clauses, True, inputs_to_region, def_but_live_out, private_to_region)
        if config.DEBUG_OPENMP >= 1:
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit after:", k, v)
            for v in clauses:
                print("vars_in_explicit clauses after:", v)
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("post get_explicit_vars:", explicit_privates)
            for k,v in vars_in_explicit_clauses.items():
                print("vars_in_explicit post:", k, v)
        blocks_in_region = get_blocks_between_start_end(self.blocks, self.blk_start, self.blk_end)
        if config.DEBUG_OPENMP >= 1:
            print(1, "blocks_in_region:", blocks_in_region)
        replace_vardict, copying_ir, copying_ir_before, lastprivate_copying = self.replace_private_vars(blocks_in_region, vars_in_explicit_clauses, explicit_privates, clauses, scope, self.loc, orig_inputs_to_region)
        assert(len(lastprivate_copying) == 0)
        before_start.extend(copying_ir_before)
        after_start.extend(copying_ir)

        priv_saves = []
        priv_restores = []
        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, live_out_copy, scope, self.loc)
        if config.DEBUG_OPENMP >= 1:
            print("clause_privates:", clause_privates, type(clause_privates))
            print("inputs_to_region:", inputs_to_region)
            print("def_but_live_out:", def_but_live_out)
            print("live_out_copy:", live_out_copy)
            print("private_to_region:", private_to_region)

        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = scope.redefine("$"+cp, self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        start_tags = [openmp_tag("DIR.OMP.PARALLEL")]
        end_tags = [openmp_tag("DIR.OMP.END.PARALLEL")]
        keep_alive = []
        self.add_explicits_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive)
        #self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = priv_saves + before_start + [or_start] + after_start + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        #eblk.body = [or_end] + priv_restores + eblk.body[:]
        eblk.body = [or_end] + priv_restores + keep_alive + eblk.body[:]

        add_enclosing_region(self.func_ir, self.body_blocks, or_start)

    def parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_clause", args, type(args), args[0])
        return val

    def unique_parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit unique_parallel_clause", args, type(args), args[0])
        assert(isinstance(val, openmp_tag))
        return val

    def teams_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit teams_clause", args, type(args), args[0])
        return val

    def num_teams_clause(self, args):
        (_, num_teams) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit num_teams_clause", args, type(args))

        return openmp_tag("QUAL.OMP.NUM_TEAMS", num_teams, load=True)

    def thread_limit_clause(self, args):
        (_, thread_limit) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit thread_limit_clause", args, type(args))

        return openmp_tag("QUAL.OMP.THREAD_LIMIT", thread_limit, load=True)

    def if_clause(self, args):
        (_, if_val) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit if_clause", args, type(args))

        return openmp_tag("QUAL.OMP.IF", if_val, load=True)

    # Don't need a rule for IF.

    def num_threads_clause(self, args):
        (_, num_threads) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit num_threads_clause", args, type(args))

        return openmp_tag("QUAL.OMP.NUM_THREADS", num_threads, load=True)

    # Don't need a rule for NUM_THREADS.
    # Don't need a rule for PARALLEL.
    # Don't need a rule for FOR.
    # Don't need a rule for parallel_for_construct.

    def parallel_for_directive(self, args):
        return self.some_for_directive(args, "DIR.OMP.PARALLEL.LOOP", "DIR.OMP.END.PARALLEL.LOOP", 2, True)

    def parallel_for_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_for_clause", args, type(args), args[0])
        return args[0]

    # Don't need a rule for for_construct.

    def for_directive(self, args):
        return self.some_for_directive(args, "DIR.OMP.LOOP", "DIR.OMP.END.LOOP", 1, False)

    def for_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return [val]
        elif isinstance(val, list):
            return val
        elif val == 'nowait':
            return openmp_tag("QUAL.OMP.NOWAIT")

    def unique_for_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit unique_for_clause", args, type(args))
        if isinstance(val, openmp_tag):
            return val
        elif val == 'ordered':
            return openmp_tag("QUAL.OMP.ORDERED", 0)

    # Don't need a rule for LINEAR.

    def linear_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit linear_clause", args, type(args), args[0])
        return args[0]

    """
    Linear_expr not in grammar
    def linear_expr(self, args):
        (_, var, step) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit linear_expr", args, type(args))
        return openmp_tag("QUAL.OMP.LINEAR", [var, step])
    """

    """
    def ORDERED(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit ordered", args, type(args))
        return "ordered"
    """

    def sched_no_expr(self, args):
        (_, kind) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit sched_no_expr", args, type(args))
        if kind == 'static':
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", 0)
        elif kind == 'dynamic':
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", 0)
        elif kind == 'guided':
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", 0)
        elif kind == 'runtime':
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", 0)

    def sched_expr(self, args):
        (_, kind, num_or_var) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit sched_expr", args, type(args), num_or_var, type(num_or_var))
        if kind == 'static':
            return openmp_tag("QUAL.OMP.SCHEDULE.STATIC", num_or_var, load=True)
        elif kind == 'dynamic':
            return openmp_tag("QUAL.OMP.SCHEDULE.DYNAMIC", num_or_var, load=True)
        elif kind == 'guided':
            return openmp_tag("QUAL.OMP.SCHEDULE.GUIDED", num_or_var, load=True)
        elif kind == 'runtime':
            return openmp_tag("QUAL.OMP.SCHEDULE.RUNTIME", num_or_var, load=True)

    def SCHEDULE(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit SCHEDULE", args, type(args))
        return "schedule"

    def schedule_kind(self, args):
        (kind,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit schedule_kind", args, type(args))
        return kind

    # Don't need a rule for STATIC.
    # Don't need a rule for DYNAMIC.
    # Don't need a rule for GUIDED.
    # Don't need a rule for RUNTIME.

    """
    def STATIC(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit STATIC", args, type(args))
        return "static"

    def DYNAMIC(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit DYNAMIC", args, type(args))
        return "dynamic"

    def GUIDED(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit GUIDED", args, type(args))
        return "guided"

    def RUNTIME(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit RUNTIME", args, type(args))
        return "runtime"
    """

    def COLON(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit COLON", args, type(args))
        return ":"

    def oslice(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit oslice", args, type(args))
        start = None
        end = None
        if args[0] != ":":
            start = args[0]
            args = args[2:]
        else:
            args = args[1:]

        if len(args) > 0:
            end = args[0]
        return slice(start, end)

    def slice_list(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit slice_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def name_slice(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit name_slice", args, type(args))
        if len(args) == 1:
            return args[0]
        else:
            return NameSlice(args[0], args[1:])

    def var_list(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit var_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

    def PLUS(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit PLUS", args, type(args))
        return "+"

    def reduction_operator(self, args):
        arg = args[0]
        if config.DEBUG_OPENMP >= 1:
            print("visit reduction_operator", args, type(args), arg, type(arg))
        if arg == "+":
            return "ADD"
        assert(0)

    def threadprivate_directive(self, args):
        raise NotImplementedError("Threadprivate currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit threadprivate_directive", args, type(args))

    def cancellation_point_directive(self, args):
        raise NotImplementedError("""Explicit cancellation points
                                 currently unsupported.""")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit cancellation_point_directive", args, type(args))

    def construct_type_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit construct_type_clause", args, type(args), args[0])
        return args[0]

    def cancel_directive(self, args):
        raise NotImplementedError("Cancel directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit cancel_directive", args, type(args))

    # Don't need a rule for ORDERED.

    def flush_directive(self, args):
        raise NotImplementedError("Flush directive currently unsupported.")
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit flush_directive", args, type(args))

    def region_phrase(self, args):
        raise NotImplementedError("No implementation for region phrase.")
        if config.DEBUG_OPENMP >= 1:
            print("visit region_phrase", args, type(args))

    def PYTHON_NAME(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit PYTHON_NAME", args, type(args), str(args))
        return str(args)

    def NUMBER(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit NUMBER", args, type(args), str(args))
        return int(args)


openmp_grammar = r"""
    openmp_statement: openmp_construct
                    | openmp_directive
    openmp_directive: barrier_directive
                    | taskwait_directive
                    | taskyield_directive
                    | flush_directive
    barrier_directive: BARRIER
    taskwait_directive: TASKWAIT
    taskyield_directive: TASKYIELD
    BARRIER: "barrier"
    TASKWAIT: "taskwait"
    TASKYIELD: "taskyield"
    taskgroup_directive: TASKGROUP
    taskgroup_construct: taskgroup_directive
    TASKGROUP: "taskgroup"
    openmp_construct: parallel_construct
                    | parallel_for_construct
                    | for_construct
                    | single_construct
                    | task_construct
                    | teams_construct
                    | target_construct
                    | target_teams_construct
                    | target_teams_distribute_parallel_for_simd_construct
                    | target_teams_distribute_parallel_for_construct
                    | target_teams_loop_construct
                    | target_enter_data_construct
                    | target_exit_data_construct
                    | distribute_construct
                    | distribute_parallel_for_construct
                    | critical_construct
                    | atomic_construct
                    | sections_construct
                    | section_construct
                    | simd_construct
                    | for_simd_construct
                    | parallel_for_simd_construct
                    | target_data_construct
                    | target_update_construct
                    | parallel_sections_construct
                    | master_construct
                    | ordered_construct
    //teams_distribute_parallel_for_simd_clause: target_clause
    //                                         | teams_distribute_parallel_for_simd_clause
    for_simd_construct: for_simd_directive
    for_simd_directive: FOR SIMD [for_simd_clause*]
    for_simd_clause: for_clause
                   | simd_clause
    parallel_for_simd_construct: parallel_for_simd_directive
    parallel_for_simd_directive: PARALLEL FOR SIMD [parallel_for_simd_clause*]
    parallel_for_simd_clause: parallel_for_clause
                            | simd_clause
    distribute_construct: distribute_directive
    distribute_directive: DISTRIBUTE [distribute_clause*]
    distribute_clause: data_privatization_clause
                     | data_privatization_in_clause
              //     | lastprivate_distribute_clause
                     | collapse_clause
                     | dist_schedule_clause
                     | allocate_clause

    distribute_parallel_for_construct: distribute_parallel_for_directive
    distribute_parallel_for_directive: DISTRIBUTE PARALLEL FOR [distribute_parallel_for_clause*]
    distribute_parallel_for_clause: if_clause
                                  | num_threads_clause
                                  | data_default_clause
                                  | data_privatization_clause
                                  | data_privatization_in_clause
                                  | data_sharing_clause
                                  | data_reduction_clause
                                  | copyin_clause
                           //     | proc_bind_clause
                                  | allocate_clause
                                  | data_privatization_out_clause
                                  | linear_clause
                                  | schedule_clause
                                  | collapse_clause
                                  | ORDERED
                                  | NOWAIT
                           //     | order_clause
                                  | dist_schedule_clause
    target_data_construct: target_data_directive
    target_data_directive: TARGET DATA [target_data_clause*]
    DATA: "data"
    ENTER: "enter"
    EXIT: "exit"
    target_enter_data_construct: target_enter_data_directive
    target_enter_data_directive: TARGET ENTER DATA [target_data_clause*]
    target_exit_data_construct: target_exit_data_directive
    target_exit_data_directive: TARGET EXIT DATA [target_data_clause*]
    target_data_clause: device_clause
                      | map_clause
                      | if_clause
                      | NOWAIT
                      | depend_with_modifier_clause
    device_clause: "device" "(" const_num_or_var ")"
    map_clause: "map" "(" [map_type ":"] var_list ")"
    map_type: ALLOC | TO | FROM | TOFROM
    TO: "to"
    FROM: "from"
    ALLOC: "alloc"
    TOFROM: "tofrom"
    parallel_sections_construct: parallel_sections_directive
    parallel_sections_directive: PARALLEL SECTIONS [parallel_sections_clause*]
    parallel_sections_clause: unique_parallel_clause
                            | data_default_clause
                            | data_privatization_clause
                            | data_privatization_in_clause
                            | data_privatization_out_clause
                            | data_sharing_clause
                            | data_reduction_clause
    sections_construct: sections_directive
    sections_directive: SECTIONS [sections_clause*]
    SECTIONS: "sections"
    sections_clause: data_privatization_clause
                   | data_privatization_in_clause
                   | data_privatization_out_clause
                   | data_reduction_clause
                   | NOWAIT
    section_construct: section_directive
    section_directive: SECTION
    SECTION: "section"
    atomic_construct: atomic_directive
    atomic_directive: ATOMIC [atomic_clause] [seq_cst_clause]
    ATOMIC: "atomic"
    atomic_clause: READ
                 | WRITE
                 | UPDATE
                 | CAPTURE
    READ: "read"
    WRITE: "write"
    UPDATE: "update"
    CAPTURE: "capture"
    seq_cst_clause: "seq_cst"
    critical_construct: critical_directive
    critical_directive: CRITICAL
    CRITICAL: "critical"
    teams_construct: teams_directive
    teams_directive: TEAMS [teams_clause*]
    target_construct: target_directive
    target_teams_distribute_parallel_for_simd_construct: target_teams_distribute_parallel_for_simd_directive
    target_teams_distribute_parallel_for_construct: target_teams_distribute_parallel_for_directive
    target_teams_loop_construct: target_teams_loop_directive
    target_teams_construct: target_teams_directive
    target_directive: TARGET [target_clause*]
    HAS_DEVICE_ADDR: "has_device_addr"
    has_device_addr_clause: HAS_DEVICE_ADDR "(" var_list ")"
    target_clause: if_clause
                 | device_clause
                 | thread_limit_clause
                 | data_privatization_clause
                 | data_privatization_in_clause
          //     | in_reduction_clause
                 | map_clause
                 | is_device_ptr_clause
                 | has_device_addr_clause
          //     | defaultmap_clause
                 | NOWAIT
                 | allocate_clause
                 | depend_with_modifier_clause
          //     | uses_allocators_clause
    teams_clause: num_teams_clause
                | thread_limit_clause
                | data_default_clause
                | data_privatization_clause
                | data_privatization_in_clause
                | data_sharing_clause
                | data_reduction_clause
                | allocate_clause
    num_teams_clause: NUM_TEAMS "(" const_num_or_var ")"
    NUM_TEAMS: "num_teams"
    thread_limit_clause: THREAD_LIMIT "(" const_num_or_var ")"
    THREAD_LIMIT: "thread_limit"

    dist_schedule_expr: DIST_SCHEDULE "(" STATIC ")"
    dist_schedule_no_expr: DIST_SCHEDULE "(" STATIC "," const_num_or_var ")"
    dist_schedule_clause: dist_schedule_expr
                        | dist_schedule_no_expr
    DIST_SCHEDULE: "dist_schedule"

    target_teams_distribute_parallel_for_simd_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR SIMD [target_teams_distribute_parallel_for_simd_clause*]
    target_teams_distribute_parallel_for_simd_clause: if_clause
                                                    | device_clause
                                                    | data_privatization_clause
                                                    | data_privatization_in_clause
                                             //     | in_reduction_clause
                                                    | map_clause
                                                    | is_device_ptr_clause
                                             //     | defaultmap_clause
                                                    | NOWAIT
                                                    | allocate_clause
                                                    | depend_with_modifier_clause
                                             //     | uses_allocators_clause
                                                    | num_teams_clause
                                                    | thread_limit_clause
                                                    | data_default_clause
                                                    | data_sharing_clause
                                                    | data_reduction_clause
                                                    | num_threads_clause
                                                    | copyin_clause
                                             //     | proc_bind_clause
                                                    | data_privatization_out_clause
                                                    | linear_clause
                                                    | schedule_clause
                                                    | collapse_clause
                                                    | ORDERED
                                             //     | order_clause
                                                    | dist_schedule_clause
                                             //     | safelen_clause
                                             //     | simdlen_clause
                                                    | aligned_clause
                                             //     | nontemporal_clause

    target_teams_distribute_parallel_for_directive: TARGET TEAMS DISTRIBUTE PARALLEL FOR [target_teams_distribute_parallel_for_clause*]
    target_teams_distribute_parallel_for_clause: if_clause
                                               | device_clause
                                               | data_privatization_clause
                                               | data_privatization_in_clause
                                        //     | in_reduction_clause
                                               | map_clause
                                               | is_device_ptr_clause
                                        //     | defaultmap_clause
                                               | NOWAIT
                                               | allocate_clause
                                               | depend_with_modifier_clause
                                        //     | uses_allocators_clause
                                               | num_teams_clause
                                               | thread_limit_clause
                                               | data_default_clause
                                               | data_sharing_clause
                                               | data_reduction_clause
                                               | num_threads_clause
                                               | copyin_clause
                                        //     | proc_bind_clause
                                               | data_privatization_out_clause
                                               | linear_clause
                                               | schedule_clause
                                               | collapse_clause
                                               | ORDERED
                                        //     | order_clause
                                               | dist_schedule_clause

    LOOP: "loop"

    target_teams_loop_directive: TARGET TEAMS LOOP [target_teams_loop_clause*]
    target_teams_loop_clause: if_clause
                            | device_clause
                            | data_privatization_clause
                            | data_privatization_in_clause
                     //     | in_reduction_clause
                            | map_clause
                            | is_device_ptr_clause
                     //     | defaultmap_clause
                            | NOWAIT
                            | allocate_clause
                            | depend_with_modifier_clause
                     //     | uses_allocators_clause
                            | num_teams_clause
                            | thread_limit_clause
                            | data_default_clause
                            | data_sharing_clause
                     //     | reduction_default_only_clause
                     //     | bind_clause
                            | collapse_clause
                            | ORDERED
                            | data_privatization_out_clause

    target_teams_directive: TARGET TEAMS [target_teams_clause*]
    target_teams_clause: if_clause
                       | device_clause
                       | data_privatization_clause
                       | data_privatization_in_clause
                //     | in_reduction_clause
                       | map_clause
                       | is_device_ptr_clause
                //     | defaultmap_clause
                       | NOWAIT
                       | allocate_clause
                       | depend_with_modifier_clause
                //     | uses_allocators_clause
                       | num_teams_clause
                       | thread_limit_clause
                       | data_default_clause
                       | data_sharing_clause
                //     | reduction_default_only_clause

    IS_DEVICE_PTR: "is_device_ptr"
    is_device_ptr_clause: IS_DEVICE_PTR "(" var_list ")"
    allocate_clause: ALLOCATE "(" allocate_parameter ")"
    ALLOCATE: "allocate"
    allocate_parameter: [const_num_or_var] var_list

    target_update_construct: target_update_directive
    target_update_directive: TARGET UPDATE target_update_clause*
    target_update_clause: motion_clause
                        | device_clause
                        | if_clause
    motion_clause: "to" "(" variable_array_section_list ")"
                 | "from" "(" variable_array_section_list ")"
    variable_array_section_list: PYTHON_NAME
                               | array_section
                               | variable_array_section_list "," PYTHON_NAME
                               | variable_array_section_list "," array_section
    array_section: PYTHON_NAME array_section_subscript
    array_section_subscript: array_section_subscript "[" [const_num_or_var] ":" [const_num_or_var] "]"
                           | array_section_subscript "[" const_num_or_var "]"
                           | "[" [const_num_or_var] ":" [const_num_or_var] "]"
                           | "[" const_num_or_var "]"
    TARGET: "target"
    TEAMS: "teams"
    DISTRIBUTE: "distribute"
    single_construct: single_directive
    single_directive: SINGLE [single_clause*]
    SINGLE: "single"
    single_clause: unique_single_clause
                 | data_privatization_clause
                 | data_privatization_in_clause
                 | NOWAIT
    unique_single_clause: copyprivate_clause
    NOWAIT: "nowait"
    master_construct: master_directive
    master_directive: "master"
    simd_construct: simd_directive
    simd_directive: SIMD [simd_clause*]
    SIMD: "simd"
    simd_clause: collapse_clause
               | aligned_clause
               | linear_clause
               | uniform_clause
               | data_reduction_clause
               | inbranch_clause
    aligned_clause: ALIGNED "(" var_list ")"
                  | ALIGNED "(" var_list ":" const_num_or_var ")"
    declare_simd_construct: declare_simd_directive_seq
    declare_simd_directive_seq: declare_simd_directive
                              | declare_simd_directive_seq declare_simd_directive
    declare_simd_directive: SIMD [declare_simd_clause*]
    declare_simd_clause: "simdlen" "(" const_num_or_var ")"
                       | aligned_clause
                       | linear_clause
                       | uniform_clause
                       | data_reduction_clause
                       | inbranch_clause
    ALIGNED: "aligned"
    inbranch_clause: INBRANCH | NOTINBRANCH
    INBRANCH: "inbranch"
    NOTINBRANCH: "notinbranch"
    uniform_clause: UNIFORM "(" var_list ")"
    UNIFORM: "uniform"
    collapse_clause: COLLAPSE "(" const_num_or_var ")"
    COLLAPSE: "collapse"
    task_construct: task_directive
    TASK: "task"
    task_directive: TASK [task_clause*]
    task_clause: unique_task_clause
               | data_sharing_clause
               | data_privatization_clause
               | data_privatization_in_clause
               | data_default_clause
    unique_task_clause: if_clause
                      | UNTIED
                      | MERGEABLE
                      | FINAL "(" const_num_or_var ")"
                      | depend_with_modifier_clause
    DEPEND: "depend"
    FINAL: "final"
    UNTIED: "untied"
    MERGEABLE: "mergeable"
    dependence_type: IN
                   | OUT
                   | INOUT
    depend_with_modifier_clause: DEPEND "(" dependence_type ":" variable_array_section_list ")"
    IN: "in"
    OUT: "out"
    INOUT: "inout"
    data_default_clause: default_shared_clause
                       | default_none_clause
    data_sharing_clause: shared_clause
    data_privatization_clause: private_clause
    data_privatization_in_clause: firstprivate_clause
    data_privatization_out_clause: lastprivate_clause
    data_clause: data_privatization_clause
               | copyprivate_clause
               | data_privatization_in_clause
               | data_privatization_out_clause
               | data_sharing_clause
               | data_default_clause
               | copyin_clause
               | data_reduction_clause
    private_clause: PRIVATE "(" var_list ")"
    PRIVATE: "private"
    copyprivate_clause: COPYPRIVATE "(" var_list ")"
    COPYPRIVATE: "copyprivate"
    firstprivate_clause: FIRSTPRIVATE "(" var_list ")"
    FIRSTPRIVATE: "firstprivate"
    lastprivate_clause: LASTPRIVATE "(" var_list ")"
    LASTPRIVATE: "lastprivate"
    shared_clause: SHARED "(" var_list ")"
    SHARED: "shared"
    copyin_clause: COPYIN "(" var_list ")"
    COPYIN: "copyin"
    REDUCTION: "reduction"
    data_reduction_clause: REDUCTION "(" reduction_operator ":" var_list ")"
    default_shared_clause: "default" "(" "shared" ")"
    default_none_clause: "default" "(" "none" ")"
    const_num_or_var: NUMBER | PYTHON_NAME
    parallel_construct: parallel_directive
    parallel_directive: PARALLEL [parallel_clause*]
    parallel_clause: unique_parallel_clause
                   | data_default_clause
                   | data_privatization_clause
                   | data_privatization_in_clause
                   | data_sharing_clause
                   | data_reduction_clause
    unique_parallel_clause: if_clause | num_threads_clause
    if_clause: IF "(" const_num_or_var ")"
    IF: "if"
    num_threads_clause: NUM_THREADS "(" const_num_or_var ")"
    NUM_THREADS: "num_threads"
    PARALLEL: "parallel"
    FOR: "for"
    parallel_for_construct: parallel_for_directive
    parallel_for_directive: PARALLEL FOR [parallel_for_clause*]
    parallel_for_clause: unique_parallel_clause
                       | unique_for_clause
                       | data_default_clause
                       | data_privatization_clause
                       | data_privatization_in_clause
                       | data_privatization_out_clause
                       | data_sharing_clause
                       | data_reduction_clause
    for_construct: for_directive
    for_directive: FOR [for_clause*]
    for_clause: unique_for_clause | data_clause | NOWAIT
    unique_for_clause: ORDERED
                     | sched_no_expr
                     | sched_expr
                     | collapse_clause
    LINEAR: "linear"
    linear_clause: LINEAR "(" var_list ":" const_num_or_var ")"
                 | LINEAR "(" var_list ")"
    sched_no_expr: SCHEDULE "(" schedule_kind ")"
    sched_expr: SCHEDULE "(" schedule_kind "," const_num_or_var ")"
    schedule_clause: sched_no_expr
                   | sched_expr
    SCHEDULE: "schedule"
    schedule_kind: STATIC | DYNAMIC | GUIDED | RUNTIME | AUTO
    STATIC: "static"
    DYNAMIC: "dynamic"
    GUIDED: "guided"
    RUNTIME: "runtime"
    AUTO: "auto"
    COLON: ":"
    oslice: [const_num_or_var] COLON [const_num_or_var]
    slice_list: oslice | slice_list "," oslice
    name_slice: PYTHON_NAME [ "[" slice_list "]" ]
    var_list: name_slice | var_list "," name_slice
    PLUS: "+"
    reduction_operator: PLUS | "\\" | "*" | "-" | "&" | "^" | "|" | "&&" | "||"
    threadprivate_directive: "threadprivate" "(" var_list ")"
    cancellation_point_directive: "cancellation point" construct_type_clause
    construct_type_clause: PARALLEL
                         | SECTIONS
                         | FOR
                         | TASKGROUP
    cancel_directive: "cancel" construct_type_clause [if_clause]
    ordered_directive: ORDERED
    ordered_construct: ordered_directive
    ORDERED: "ordered"
    flush_directive: "flush" "(" var_list ")"

    region_phrase: "(" PYTHON_NAME ")"
    PYTHON_NAME: /[a-zA-Z_]\w*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
    """

"""
    name_slice: PYTHON_NAME [ "[" slice ["," slice]* "]" ]
    openmp_construct: parallel_construct
                    | target_teams_distribute_construct
                    | teams_distribute_parallel_for_simd_construct
                    | target_teams_distribute_parallel_for_construct
                    | target_teams_construct
                    | target_teams_distribute_simd_construct
                    | teams_distribute_parallel_for_construct
                    | teams_construct
                    | distribute_parallel_for_construct
                    | distribute_parallel_for_simd_construct
                    | distribute_construct
                    | distribute_simd_construct
                    | teams_distribute_construct
                    | teams_distribute_simd_construct
"""

openmp_parser = Lark(openmp_grammar, start='openmp_statement')
var_collector_parser = Lark(openmp_grammar, start='openmp_statement')

def remove_ssa_callback(var, unused):
    assert isinstance(var, ir.Var)
    new_var = ir.Var(var.scope, var.unversioned_name, var.loc)
    return new_var


def remove_ssa_from_func_ir(func_ir):
    typed_passes.PreLowerStripPhis()._strip_phi_nodes(func_ir)
#    new_func_ir = typed_passes.PreLowerStripPhis()._strip_phi_nodes(func_ir)
#    func_ir.blocks = new_func_ir.blocks
    visit_vars(func_ir.blocks, remove_ssa_callback, None)
    func_ir._definitions = build_definitions(func_ir.blocks)


def _add_openmp_ir_nodes(func_ir, blocks, blk_start, blk_end, body_blocks, extra, state):
    """Given the starting and ending block of the with-context,
    replaces the head block with a new block that has the starting
    openmp ir nodes in it and adds the ending openmp ir nodes to
    the end block.
    """
    # First check for presence of required libraries.
    if library_missing:
        if iomplib is None:
            print("OpenMP runtime library could not be found.")
            print("Make sure that libomp.so or libiomp5.so is in your library path or")
            print("specify the location of the OpenMP runtime library with the")
            print("NUMBA_OMP_LIB environment variables.")
            sys.exit(-1)

        if omptargetlib is None:
            print("OpenMP target runtime library could not be found.")
            print("Make sure that libomptarget.so or")
            print("specify the location of the OpenMP runtime library with the")
            print("NUMBA_OMPTARGET_LIB environment variables.")
            sys.exit(-1)

    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    sblk.body = sblk.body[1:]

    args = extra["args"]
    arg = args[0]
    # If OpenMP argument is not a constant or not a string then raise exception
    if not isinstance(arg, ir.Const):
        raise NonconstantOpenmpSpecification(f"Non-constant OpenMP specification at line {arg.loc}")
    if not isinstance(arg.value, str):
        raise NonStringOpenmpSpecification(f"Non-string OpenMP specification at line {arg.loc}")

    parse_res = openmp_parser.parse(arg.value)
    if config.DEBUG_OPENMP >= 1:
        print("args:", args, type(args))
        print("arg:", arg, type(arg), arg.value, type(arg.value))
        print(parse_res.pretty())
    visitor = OpenmpVisitor(func_ir, blocks, blk_start, blk_end, body_blocks, loc, state)
    try:
        visitor.transform(parse_res)
    except VisitError as e:
        raise e.__context__
        if isinstance(e.__context__, UnspecifiedVarInDefaultNone):
            print(str(e.__context__))
            raise e.__context__
        else:
            print("Internal error for OpenMp pragma '{}'".format(arg.value), e.__context__, type(e.__context__))
        sys.exit(-1)
    except Exception as f:
        print("generic transform exception")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Internal error for OpenMp pragma '{}'".format(arg.value))
        sys.exit(-2)
    except:
        print("fallthrough exception")
        print("Internal error for OpenMp pragma '{}'".format(arg.value))
        sys.exit(-3)
    assert(blocks is visitor.blocks)

omp_runtime_funcs = [
    ("omp_set_num_threads", ("void", "types.void"), [("int", "types.int32", "num_threads")]),
    ("omp_get_thread_num", ("int", "types.int32"), []),
    ("omp_get_num_threads", ("int", "types.int32"), []),
    ("omp_get_wtime", ("double", "types.float64"), []),
    ("omp_set_dynamic", ("void", "types.void"), [("int", "types.int32", "num_threads")]),
    ("omp_set_nested", ("void", "types.void"), [("int", "types.int32", "nested")]),
    ("omp_set_max_active_levels", ("void", "types.void"), [("int", "types.int32", "levels")]),
    ("omp_get_max_active_levels", ("int", "types.int32"), []),
    ("omp_get_max_threads", ("int", "types.int32"), []),
    ("omp_get_num_procs", ("int", "types.int32"), []),
    ("omp_in_parallel", ("int", "types.int32"), []),
    ("omp_get_thread_limit", ("int", "types.int32"), []),
    ("omp_get_supported_active_levels", ("int", "types.int32"), []),
    ("omp_get_level", ("int", "types.int32"), []),
    ("omp_get_active_level", ("int", "types.int32"), []),
    ("omp_get_ancestor_thread_num", ("int", "types.int32"), [("int", "types.int32", "level")]),
    ("omp_get_team_size", ("int", "types.int32"), [("int", "types.int32", "level")]),
    ("omp_in_final", ("int", "types.int32"), []),
    ("omp_get_proc_bind", ("int", "types.int32"), []),
    ("omp_get_num_places", ("int", "types.int32"), []),
    ("omp_get_place_num_procs", ("int", "types.int32"), [("int", "types.int32", "place_num")]),
    ("omp_get_place_num", ("int", "types.int32"), []),
    ("omp_set_default_device", ("int", "types.int32"), [("int", "types.int32", "device_num")]),
    ("omp_get_default_device", ("int", "types.int32"), []),
    ("omp_get_num_devices", ("int", "types.int32"), []),
    ("omp_get_device_num", ("int", "types.int32"), []),
    ("omp_get_team_num", ("int", "types.int32"), []),
    ("omp_get_num_teams", ("int", "types.int32"), []),
    ("omp_is_initial_device", ("int", "types.int32"), []),
    ("omp_get_initial_device", ("int", "types.int32"), []),
]

# For all the OpenMP runtime functions in the list above,
# dynamically create a pure Python function that invokes
# the runtime functions with cffi.  Also generates a
# Numba overload of that function that calls it as an
# external function.
for fname, retinfo, arginfo in omp_runtime_funcs:
    def form_argstr(retinfo, arginfo):
        return ",".join([x[2] for x in arginfo])

    def form_cdef_args(retinfo, arginfo):
        return ",".join([x[0] + " " + x[2] for x in arginfo])

    def form_overload_argstr(retinfo, arginfo):
        return ",".join([x[1] for x in arginfo])

    argstr = form_argstr(retinfo, arginfo)
    cdef_args = form_cdef_args(retinfo, arginfo)
    overload_argstr = form_overload_argstr(retinfo, arginfo)

    fdef = f"""def {fname}({argstr}):
    ffi = FFI()
    ffi.cdef('{retinfo[0]} {fname}({cdef_args});')
    C = ffi.dlopen(None)
    return C.{fname}({argstr})
    """

    ldict = {}
    gdict = globals()

    #print("fdef:", fdef)
    exec(fdef, gdict, ldict)
    fout = ldict[fname]
    gdict[fname] = fout

    odef = f"""def ol_{fname}({argstr}):
    fnty = types.ExternalFunction("{fname}", {retinfo[1]}({overload_argstr}))
    def impl({argstr}):
        return fnty({argstr})
    return impl
    """
    #print("odef:", odef)
    exec(odef, gdict, ldict)
    oout = ldict[f"ol_{fname}"]
    overload(fout)(oout)
#    overload(fout, target="cuda")(oout)
