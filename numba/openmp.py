from numba.core.withcontexts import WithContext
from lark import Lark, Transformer
from lark.exceptions import VisitError
from numba.core.ir_utils import (
    get_call_table,
    mk_unique_var,
    dump_block,
    dump_blocks,
    dprint_func_ir,
    replace_vars,
    apply_copy_propagate_extensions,
    visit_vars_extensions,
    remove_dels,
    visit_vars_inner,
    get_name_var_table,
    replace_var_names,
)
from numba.core.analysis import compute_cfg_from_blocks, compute_use_defs, compute_live_map, find_top_level_loops
from numba.core import ir, config, types, typeinfer, cgutils, compiler, transforms, bytecode
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

#----------------------------------------------------------------------------------------------

class NameSlice:
    def __init__(self, name, the_slice):
        self.name = name
        self.the_slice = the_slice

    def __str__(self):
        return "NameSlice(" + str(self.name) + "," + str(self.the_slice) + ")"


class openmp_tag(object):
    def __init__(self, name, arg=None, load=False):
        self.name = name
        self.arg = arg
        self.load = load
        self.loaded_arg = None
        self.xarginfo = []

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

    def arg_to_str(self, x, lowerer):
        if config.DEBUG_OPENMP >= 1:
            print("arg_to_str:", x, type(x), self.load, type(self.load))
        if isinstance(x, NameSlice):
            if config.DEBUG_OPENMP >= 2:
                print("nameslice found:", x)
            x = x.name
        if isinstance(x, ir.Var):
            # Make sure the var referred to has been alloc'ed already.
            lowerer._alloca_var(x.name, lowerer.fndesc.typemap[x.name])
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
            xtyp = lowerer.fndesc.typemap[x]
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
        elif isinstance(x, int):
            decl = "i32 " + str(x)
        else:
            print("unknown arg type:", x, type(x))

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

        decl = ",".join([self.arg_to_str(x, lowerer) for x in arg_list])

        return '"' + self.name + '"(' + decl + ')'

    def replace_vars_inner(self, var_dict):
        if isinstance(self.arg, ir.Var):
            self.arg = replace_vars_inner(self.arg, var_dict)

    def add_to_use_set(self, use_set):
        if isinstance(self.arg, ir.Var):
            use_set.add(self.arg.name)

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
    cgutils.push_alloca_callbacks(callback, data)
    if not hasattr(builder, '_lowerer_push_alloca_callbacks'):
        builder._lowerer_push_alloca_callbacks = 0
    builder._lowerer_push_alloca_callbacks += 1

def pop_alloca_callback(lowerer, builder):
    cgutils.pop_alloca_callbacks()
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

    """
    def __new__(cls, *args, **kwargs):
        #breakpoint()
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
        has_update = False
        for alloca_instr, typ in self.alloca_queue:
            has_update = self.process_one_alloca(alloca_instr, typ) or has_update
        if has_update:
            self.update_tags()

    def process_one_alloca(self, alloca_instr, typ):
        avar = alloca_instr.name
        if config.DEBUG_OPENMP >= 1:
            print("openmp_region_start process_one_alloca:", id(self), alloca_instr, avar, typ, type(alloca_instr))

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
            self.tags.append(openmp_tag("QUAL.OMP.PRIVATE", alloca_instr)) # is FIRSTPRIVATE right here?
            #self.tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", alloca_instr)) # is FIRSTPRIVATE right here?
            self.tag_vars.add(avar)
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
        context = lowerer.context
        builder = lowerer.builder
        library = lowerer.library
        self.block = builder.block
        self.builder = builder
        self.lowerer = lowerer
        if config.DEBUG_OPENMP >= 1:
            print("region ids lower:block", id(self), self, id(self.block), self.block, type(self.block), self.tags, len(self.tags), "builder_id:", id(self.builder), "block_id:", id(self.block))
            for k,v in lowerer.func_ir.blocks.items():
                print("block post copy:", k, id(v), id(v.body))

        if config.DEBUG_OPENMP >= 2:
            for otag in self.tags:
                print("otag:", otag, type(otag.arg))

        # Remove LLVM vars that might have been added if this is an OpenMP
        # region inside a target region.
        count_alloca_instr = len(list(filter(lambda x: isinstance(x.arg, lir.instructions.AllocaInstr), self.tags)))
        assert(count_alloca_instr == 0)
        #self.tags = list(filter(lambda x: not isinstance(x.arg, lir.instructions.AllocaInstr), self.tags))
        if config.DEBUG_OPENMP >= 1:
            print("after LLVM tag filter", self.tags, len(self.tags))

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

        target_num = self.has_target()
        tgv = None
        if target_num is not None and self.target_copy != True:
            #from numba.cuda import descriptor as cuda_descriptor, compiler as cuda_compiler
            from numba.core.compiler import Compiler, Flags
            #builder.module.device_triples = "spir64"
            if config.DEBUG_OPENMP >= 1:
                print("openmp start region lower has target", type(lowerer.func_ir))
            #func_ir = lowerer.func_ir.deepcopy()
            func_ir = lowerer.func_ir.copy()
            for k,v in lowerer.func_ir.blocks.items():
                print("region ids block post copy:", k, id(v), id(func_ir.blocks[k]), id(v.body), id(func_ir.blocks[k].body))

            """
            var_table = get_name_var_table(func_ir.blocks)
            new_var_dict = {}
            for name, var in var_table.items():
                new_var_dict[name] = mk_unique_var(name)
            replace_var_names(func_ir.blocks, new_var_dict)
            """

            remove_dels(func_ir.blocks)

            dprint_func_ir(func_ir, "func_ir after remove_dels")

            def fixup_openmp_pairs(blocks):
                start_dict = {}
                end_dict = {}
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

                for start_id, blockindex in start_dict.items():
                    start_block, sbindex = blockindex

                    end_block_index = end_dict[start_id]
                    end_block, ebindex = end_block_index

                    if config.DEBUG_OPENMP >= 2:
                        start_pre_copy = blocks[start_block].body[sbindex]
                        end_pre_copy = blocks[end_block].body[ebindex]

                    blocks[start_block].body[sbindex] = copy.copy(blocks[start_block].body[sbindex])
                    blocks[end_block].body[ebindex] = copy.copy(blocks[end_block].body[ebindex])
                    start_region = blocks[start_block].body[sbindex]
                    start_region.builder = None
                    start_region.block = None
                    start_region.lowerer = None
                    start_region.target_copy = True
                    start_region.tags = copy.deepcopy(start_region.tags)
                    if start_region.has_target() == target_num:
                        start_region.tags.append(openmp_tag("OMP.DEVICE"))
                    end_region = blocks[end_block].body[ebindex]
                    assert(start_region.omp_region_var is None)
                    assert(len(start_region.alloca_queue) == 0)
                    end_region.start_region = start_region
                    start_region.end_region = end_region
                    if config.DEBUG_OPENMP >= 2:
                        print(f"region ids fixup start: {id(start_pre_copy)}->{id(start_region)} end: {id(end_pre_copy)}->{id(end_region)}")

            fixup_openmp_pairs(func_ir.blocks)
            state = lowerer.state
            fndesc = lowerer.fndesc
            if config.DEBUG_OPENMP >= 2:
                print("context:", context, type(context))
                print("targetctx:", targetctx, type(targetctx))
                print("state:", state, dir(state))
                print("fndesc:", fndesc, type(fndesc))
            dprint_func_ir(func_ir, "target func_ir")
            internal_codegen = targetctx._internal_codegen
            #target_module = internal_codegen._create_empty_module("openmp.target")

            start_block, end_block = find_target_start_end(func_ir, target_num)
            cfg = compute_cfg_from_blocks(func_ir.blocks)

            remove_openmp_nodes_from_target = False
            if not remove_openmp_nodes_from_target:
                end_target_node = func_ir.blocks[end_block].body[0]

            if config.DEBUG_OPENMP >= 1:
                print("start_block:", start_block)
                print("end_block:", end_block)
            blocks_in_region = [start_block]
            def add_in_region(cfg, blk, blocks_in_region, end_block):
                for out_blk, _ in cfg.successors(blk):
                    if out_blk != end_block and out_blk not in blocks_in_region:
                        blocks_in_region.append(out_blk)
                        add_in_region(cfg, out_blk, blocks_in_region, end_block)

            add_in_region(cfg, start_block, blocks_in_region, end_block)
            if config.DEBUG_OPENMP >= 1:
                print("blocks_in_region:", blocks_in_region)
            ins, outs = transforms.find_region_inout_vars(
                blocks=func_ir.blocks,
                livemap=func_ir.variable_lifetime.livemap,
                callfrom=start_block,
                returnto=end_block,
                body_block_ids=blocks_in_region
            )
            outline_arg_typs = tuple([typemap[x] for x in ins])
            if config.DEBUG_OPENMP >= 1:
                print("ins:", ins)
                print("outs:", outs)
                print("args:", state.args)
                print("rettype:", state.return_type, type(state.return_type))
                print("outline_arg_typs:", outline_arg_typs)
            region_info = transforms._loop_lift_info(loop=None,
                                                     inputs=ins,
                                                     outputs=outs,
                                                     callfrom=start_block,
                                                     returnto=end_block)

            region_blocks = dict((k, func_ir.blocks[k]) for k in blocks_in_region)
            #region_blocks = dict((k, func_ir.blocks[k].copy()) for k in blocks_in_region)
            transforms._loop_lift_prepare_loop_func(region_info, region_blocks)

            if remove_openmp_nodes_from_target:
                region_blocks[start_block].body = region_blocks[start_block].body[1:]
                #region_blocks[end_block].body = region_blocks[end_block].body[1:]

            #region_blocks = copy.copy(region_blocks)
            #region_blocks = copy.deepcopy(region_blocks)
            # transfer_scope?
            # core/untyped_passes/versioning_loop_bodies

            outlined_ir = func_ir.derive(blocks=region_blocks,
                                         arg_names=tuple(ins),
                                         arg_count=len(ins),
                                         force_non_generator=True)
            fparts = outlined_ir.func_id.func_qualname.split('.')
            fparts[-1] = "device" + fparts[-1]
            outlined_ir.func_id.func_qualname = ".".join(fparts)
            outlined_ir.func_id.func_name = fparts[-1]
            uid = next(bytecode.FunctionIdentity._unique_ids)
            outlined_ir.func_id.unique_name = '{}${}'.format(outlined_ir.func_id.func_qualname, uid)
            print("outlined_ir:", type(outlined_ir), type(outlined_ir.func_id), fparts)

            #cuda_typingctx = cuda_descriptor.cuda_target.typing_context
            #cuda_targetctx = cuda_descriptor.cuda_target.target_context
            flags = Flags()
            #flags = cuda_compiler.CUDAFlags()
            # Do not compile (generate native code), just lower (to LLVM)
            flags.no_compile = True
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True
            # What to do here?
            flags.forceinline = True
            #flags.fastmath = True
            class OnlyLower(compiler.CompilerBase):
                def define_pipelines(self):
                    pms = []
                    if not self.state.flags.force_pyobject:
                        pms.append(compiler.DefaultPassBuilder.define_nopython_lowering_pipeline(self.state))
                    return pms

            state_copy = copy.copy(state)
            state_copy.typemap = copy.copy(typemap)

            for var_in in ins:
                arg_name = "arg." + var_in
                state_copy.typemap.pop(arg_name, None)
                state_copy.typemap[arg_name] = typemap[var_in]

            last_block = outlined_ir.blocks[end_block]
            if not remove_openmp_nodes_from_target:
                last_block.body = [end_target_node] + last_block.body[:]

            assert(isinstance(last_block.body[-1], ir.Return))
            state_copy.typemap[last_block.body[-1].value.name] = types.containers.Tuple(())

            if config.DEBUG_OPENMP >= 1:
                print("outlined_ir:", outlined_ir, type(outlined_ir))
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

#            breakpoint()
            cres = compiler.compile_ir(typingctx,
                                       targetctx,
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
            cres_library = cres.library
            if config.DEBUG_OPENMP >= 2:
                print("cres_library:", type(cres_library))
                sys.stdout.flush()
            #breakpoint()
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

            """
            target_elf = cres_library._get_compiled_object()
            if config.DEBUG_OPENMP >= 2:
                print("target_elf:", type(target_elf))
                sys.stdout.flush()
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
        tags_to_include = self.tags
        #tags_to_include = list(filter(lambda x: x.name != "DIR.OMP.TARGET", tags_to_include))
        self.filtered_tag_length = len(tags_to_include)
        if config.DEBUG_OPENMP >= 1:
            print("filtered_tag_length:", self.filtered_tag_length)
        print("FIX FIX FIX....this works during testing but not in target is combined with other options.  We need to remove all the target related options and then if nothing is left we can skip adding this region.")
        if len(tags_to_include) > 0:
            if config.DEBUG_OPENMP >= 1:
                print("push_alloca_callbacks")

            push_alloca_callback(lowerer, openmp_region_alloca, self, builder)
            tag_str = openmp_tag_list_to_str(tags_to_include, lowerer, True)
            pre_fn = builder.module.declare_intrinsic('llvm.directive.region.entry', (), fnty)
            assert(self.omp_region_var is None)
            self.omp_region_var = builder.call(pre_fn, [], tail=False, tags=tag_str)
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


# Callback for ir_extension_usedefs
def openmp_region_start_defs(region, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
#    for tag in region.tags:
#        tag.add_to_use_set(use_set)
    return _use_defs_result(usemap=use_set, defmap=def_set)

def openmp_region_end_defs(region, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()
#    for tag in region.tags:
#        tag.add_to_use_set(use_set)
    return _use_defs_result(usemap=use_set, defmap=def_set)

# Extend usedef analysis to suport openmp_region_start/end nodes.
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


class _OpenmpContextType(WithContext):
    is_callable = True

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end,
                         body_blocks, dispatcher_factory, extra, state=None, flags=None):
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


openmp_context = _OpenmpContextType()


def is_internal_var(var):
    # Determine if a var is a Python var or an internal Numba var.
    if var.is_temp:
        return True
    return var.unversioned_name != var.name


def remove_ssa(var_name, scope, loc):
    # Get the base name of a variable, removing the SSA extension.
    var = ir.Var(scope, var_name, loc)
    return var.unversioned_name


def has_user_defined_var(the_set):
    for x in the_set:
        if not x.startswith("$"):
            return True
    return False


def get_user_defined_var(the_set):
    ret = set()
    for x in the_set:
        if not x.startswith("$"):
            ret.add(x)
    return ret


unique = 0
def get_unique():
    global unique
    ret = unique
    unique += 1
    return ret


def is_private(x):
    return x in ["QUAL.OMP.PRIVATE", "QUAL.OMP.FIRSTPRIVATE", "QUAL.OMP.LASTPRIVATE"]


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
        if config.DEBUG_OPENMP >= 2:
            print("remove_explicit:", varset, vars_in_explicit_clauses)
        diff = set()
        # For each variable inthe set.
        for v in varset:
            # Get the non-SSA form.
            flat = remove_ssa(v, scope, loc)
            # Skip non-SSA introduced variables (i.e., Python vars).
            if flat == v:
                continue
            if config.DEBUG_OPENMP >= 2:
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
        if config.DEBUG_OPENMP >= 2:
            print("remove_explicit:", varset)

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
                    if isinstance(carg, str):
                        ret[carg] = c
                        if is_private(c.name):
                            privates.append(carg)
        return ret, privates

    def get_clause_privates(self, clauses, def_but_live_out, scope, loc):
        # Get all the private clauses from the whole set of clauses.
        private_clauses_vars = [x.arg for x in clauses if x.name == "QUAL.OMP.PRIVATE" or x.name == "QUAL.OMP.FIRSTPRIVATE"]
        ret = {}
        # Get a mapping of vars in private clauses to the SSA version of variable exiting the region.
        for lo in def_but_live_out:
            without_ssa = remove_ssa(lo, scope, loc)
            if without_ssa in private_clauses_vars:
                ret[without_ssa] = lo
        return ret

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

    def flatten(self, incoming_clauses):
        clauses = []
        for clause in incoming_clauses:
            if config.DEBUG_OPENMP >= 1:
                print("clause:", clause, type(clause))
            if isinstance(clause, openmp_tag):
                clauses.append(clause)
            elif isinstance(clause, list):
                clauses.extend(remove_indirections(clause))
            else:
                assert(0)
        return clauses

    def replace_private_vars(self, blocks, all_explicits, explicit_privates, clauses, scope, loc):
        replace_vardict = {}
        # Generate a new Numba privatized variable for each openmp private variable.
        for exp_priv in explicit_privates:
            replace_vardict[exp_priv] = ir.Var(scope, exp_priv + ".privatized." + str(get_unique()), loc)
        # Get all the blocks in this openmp region and replace the original variable with the privatized one.
        replace_vars({k: v for k, v in self.blocks.items() if k in blocks}, replace_vardict)
#        import pdb
#        pdb.set_trace()
        new_shared_clauses = []
        copying_ir = []
        copying_ir_before = []
        def do_copy(orig_name, private_name):
            g_copy_var = ir.Var(scope, mk_unique_var("$copy_g_var"), loc)
            g_copy = ir.Global("openmp_copy", openmp_copy, loc)
            #g_copy = ir.Global("numba", numba, loc)
            g_copy_assign = ir.Assign(g_copy, g_copy_var, loc)

            """
            attr_var1 = ir.Var(scope, mk_unique_var("$copy_attr_var"), loc)
            attr_getattr1 = ir.Expr.getattr(g_copy_var, 'openmp', loc)
            attr_assign1 = ir.Assign(attr_getattr1, attr_var1, loc)

            attr_var2 = ir.Var(scope, mk_unique_var("$copy_attr_var"), loc)
            attr_getattr2 = ir.Expr.getattr(g_copy_var, 'openmp_copy', loc)
            attr_assign2 = ir.Assign(attr_getattr2, attr_var2, loc)
            """

            #copy_call = ir.Expr.call(attr_var2, [orig_name], (), loc)
            copy_call = ir.Expr.call(g_copy_var, [orig_name], (), loc)
            copy_assign = ir.Assign(copy_call, private_name, loc)
            return [g_copy_assign, copy_assign]
            #return [g_copy_assign, attr_assign1, attr_assign2, copy_assign]

        for c in clauses:
            if isinstance(c.arg, str) and c.arg in replace_vardict:
                del all_explicits[c.arg]
                if c.name == "QUAL.OMP.FIRSTPRIVATE":
                    new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", c.arg))
                    all_explicits[c.arg] = new_shared_clauses[-1]
                    copying_ir.extend(do_copy(ir.Var(scope, c.arg, loc), replace_vardict[c.arg]))
                    # This one doesn't really do anything except avoid a Numba decref error for arrays.
                    copying_ir_before.append(ir.Assign(ir.Var(scope, c.arg, loc), replace_vardict[c.arg], loc))
                c.arg = replace_vardict[c.arg]
                all_explicits[c.arg] = c
            elif isinstance(c.arg, list):
                for i in range(len(c.arg)):
                    carg = c.arg[i]
                    if isinstance(carg, str) and carg in replace_vardict:
                        del all_explicits[carg]
                        if c.name == "QUAL.OMP.FIRSTPRIVATE":
                            new_shared_clauses.append(openmp_tag("QUAL.OMP.SHARED", carg))
                            all_explicits[carg] = new_shared_clauses[-1]
                            copying_ir.extend(do_copy(ir.Var(scope, carg, loc), replace_vardict[carg]))
                            # This one doesn't really do anything except avoid a Numba decref error for arrays.
                            copying_ir_before.append(ir.Assign(ir.Var(scope, carg, loc), replace_vardict[carg], loc))
                        newcarg = replace_vardict[carg].name
                        c.arg[i] = newcarg
                        all_explicits[newcarg] = c
        clauses.extend(new_shared_clauses)
        return replace_vardict, copying_ir, copying_ir_before

    def some_for_directive(self, args, main_start_tag, main_end_tag, first_clause, gen_shared):
        sblk = self.blocks[self.blk_start]
        scope = sblk.scope
        eblk = self.blocks[self.blk_end]

        clauses = []
        default_shared = True
        if config.DEBUG_OPENMP >= 1:
            print("some_for_directive", self.body_blocks)
        incoming_clauses = [remove_indirections(x) for x in args[first_clause:]]
        # Process all the incoming clauses which can be in singular or list form
        # and flatten them to a list of openmp_tags.
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
                assert(0)
        if config.DEBUG_OPENMP >= 1:
            print("visit", main_start_tag, args, type(args), default_shared)
            for clause in clauses:
                print("post-process clauses:", clause)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses), explicit_privates)

        before_start = []
        after_start = []
        start_tags = [ openmp_tag(main_start_tag) ]
        end_tags   = [ openmp_tag(main_end_tag) ]

        call_table, _ = get_call_table(self.blocks)
        cfg = compute_cfg_from_blocks(self.blocks)
        usedefs = compute_use_defs(self.blocks)
        live_map = compute_live_map(cfg, self.blocks, usedefs.usemap, usedefs.defmap)
        all_loops = cfg.loops()
        if config.DEBUG_OPENMP >= 1:
            print("all_loops:", all_loops)
        loops = {}
        # Find the outer-most loop in this OpenMP region.
        for k, v in all_loops.items():
            if v.header >= self.blk_start and v.header <= self.blk_end:
                loops[k] = v
        loops = list(find_top_level_loops(cfg, loops=loops))

        if config.DEBUG_OPENMP >= 1:
            print("loops:", loops)
        assert(len(loops) == 1)

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
        if config.DEBUG_OPENMP >= 1:
            print("loop_blocks_for_io:", loop_blocks_for_io, entry, exit)

        replace_vardict, copying_ir, copying_ir_before = self.replace_private_vars(loop_blocks_for_io_minus_entry, vars_in_explicit_clauses, explicit_privates, clauses, scope, self.loc)
        before_start.extend(copying_ir_before)
        after_start.extend(copying_ir)

        found_loop = False
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
                    found_loop = True
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
                        print("loop_index:", loop_index)
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
                                latest_index = phinst.target.name
                                break
                    if config.DEBUG_OPENMP >= 1:
                        print("latest_index:", latest_index)
                    new_index_clause = openmp_tag("QUAL.OMP.PRIVATE", ir.Var(loop_index.scope, latest_index, inst.loc))
                    clauses.append(new_index_clause)
                    vars_in_explicit_clauses[latest_index] = new_index_clause

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

                    omp_lb_var = ir.Var(loop_index.scope, mk_unique_var("$omp_lb"), inst.loc)
                    before_start.append(ir.Assign(ir.Const(0, inst.loc), omp_lb_var, inst.loc))

                    omp_iv_var = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                    #before_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))
                    after_start.append(ir.Assign(omp_lb_var, omp_iv_var, inst.loc))

                    omp_ub_var = ir.Var(loop_index.scope, mk_unique_var("$omp_ub"), inst.loc)
                    omp_ub_expr = ir.Expr.itercount(getiter_inst.target, inst.loc)
                    before_start.append(ir.Assign(omp_ub_expr, omp_ub_var, inst.loc))

                    const1_var = ir.Var(loop_index.scope, mk_unique_var("$const1"), inst.loc)
                    start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", const1_var))
                    const1_assign = ir.Assign(ir.Const(1, inst.loc), const1_var, inst.loc)
                    before_start.append(const1_assign)
                    count_add_1 = ir.Expr.binop(operator.sub, omp_ub_var, const1_var, inst.loc)
                    before_start.append(ir.Assign(count_add_1, omp_ub_var, inst.loc))

#                    before_start.append(ir.Print([omp_ub_var], None, inst.loc))

                    omp_start_var = ir.Var(loop_index.scope, mk_unique_var("$omp_start"), inst.loc)
                    if start == 0:
                        start = ir.Const(start, inst.loc)
                    before_start.append(ir.Assign(start, omp_start_var, inst.loc))

                    gen_ssa = False

                    # ---------- Create latch block -------------------------------
                    if gen_ssa:
                        latch_iv = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                    else:
                        latch_iv = omp_iv_var

                    latch_block = ir.Block(scope, inst.loc)
                    const1_var = ir.Var(loop_index.scope, mk_unique_var("$const1"), inst.loc)
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
                        self.blocks[bb].body[-1] = ir.Jump(latch_block_num, inst.loc)
                    # -------------------------------------------------------------

                    # ---------- Header Manipulation ------------------------------
                    ssa_code = []
                    if gen_ssa:
                        ssa_var = ir.Var(loop_index.scope, mk_unique_var("$omp_iv"), inst.loc)
                        ssa_phi = ir.Expr.phi(inst.loc)
                        ssa_phi.incoming_values = [latch_iv, omp_iv_var]
                        ssa_phi.incoming_blocks = [latch_block_num, entry_pred_label]
                        ssa_inst = ir.Assign(ssa_phi, ssa_var, inst.loc)
                        ssa_code.append(ssa_inst)

                    step_var = ir.Var(loop_index.scope, mk_unique_var("$step_var"), inst.loc)
                    step_assign = ir.Assign(ir.Const(step, inst.loc), step_var, inst.loc)
                    scale_var = ir.Var(loop_index.scope, mk_unique_var("$scale"), inst.loc)
                    fake_iternext = ir.Assign(ir.Const(0, inst.loc), iternext_inst.target, inst.loc)
                    fake_second = ir.Assign(ir.Const(0, inst.loc), pair_second_inst.target, inst.loc)
                    scale_assign = ir.Assign(ir.Expr.binop(operator.mul, step_var, omp_iv_var, inst.loc), scale_var, inst.loc)
                    unnormalize_iv = ir.Assign(ir.Expr.binop(operator.add, omp_start_var, scale_var, inst.loc), loop_index, inst.loc)
                    cmp_var = ir.Var(loop_index.scope, mk_unique_var("$cmp"), inst.loc)
                    if gen_ssa:
                        iv_lte_ub = ir.Assign(ir.Expr.binop(operator.le, ssa_var, omp_ub_var, inst.loc), cmp_var, inst.loc)
                    else:
                        iv_lte_ub = ir.Assign(ir.Expr.binop(operator.le, omp_iv_var, omp_ub_var, inst.loc), cmp_var, inst.loc)
                    old_branch = header_block.body[-1]
                    new_branch = ir.Branch(cmp_var, old_branch.truebr, old_branch.falsebr, old_branch.loc)
                    body_label = old_branch.truebr
                    first_body_block = self.blocks[body_label]
                    new_end = [iv_lte_ub, new_branch]
                    if False:
                        str_var = ir.Var(loop_index.scope, mk_unique_var("$str_var"), inst.loc)
                        str_const = ir.Const("header1:", inst.loc)
                        str_assign = ir.Assign(str_const, str_var, inst.loc)
                        new_end.append(str_assign)
                        str_print = ir.Print([str_var, omp_start_var, omp_iv_var], None, inst.loc)
                        new_end.append(str_print)

                    # Prepend original contents of header into the first body block minus the comparison
                    first_body_block.body = [fake_iternext, fake_second, step_assign, scale_assign, unnormalize_iv] + header_block.body[:-1] + first_body_block.body

                    header_block.body = new_end
                    #header_block.body = ssa_code + [fake_iternext, fake_second, unnormalize_iv] + header_block.body[:-1] + new_end

                    # -------------------------------------------------------------

                    #const_start_var = ir.Var(loop_index.scope, mk_unique_var("$const_start"), inst.loc)
                    #before_start.append(ir.Assign(ir.Const(0, inst.loc), const_start_var, inst.loc))
                    #start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", const_start_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", omp_iv_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", omp_ub_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_lb_var.name))
                    start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", omp_start_var.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.IV", loop_index.name))
                    #start_tags.append(openmp_tag("QUAL.OMP.NORMALIZED.UB", size_var.name))

                    keep_alive = []
                    # Do an analysis to get variable use information coming into and out of the region.
                    inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(loop_blocks_for_io)

                    # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
                    clause_privates = self.get_clause_privates(clauses, def_but_live_out, scope, self.loc)
                    priv_saves = []
                    priv_restores = []
                    # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
                    # before entering the region and then restore it afterwards but we have to restore it to the SSA
                    # version of the variable at that point.
                    for cp in clause_privates:
                        cpvar = ir.Var(scope, cp, self.loc)
                        cplovar = ir.Var(scope, clause_privates[cp], self.loc)
                        save_var = ir.Var(scope, mk_unique_var("$"+cp), self.loc)
                        priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
                        priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

                    # Remove variables the user explicitly added to a clause from the auto-determined variables.
                    # This will also treat SSA forms of vars the same as their explicit Python var clauses.
                    self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

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

                    self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, gen_shared, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

                    or_start = openmp_region_start(start_tags, 0, self.loc)
                    or_end   = openmp_region_end(or_start, end_tags, self.loc)

                    #new_header_block.body = [or_start] + before_start + new_header_block.body[:]
                    entry_pred.body = entry_pred.body[:-1] + priv_saves + before_start + [or_start] + after_start + [entry_pred.body[-1]]
                    #entry_block.body = [or_start] + before_start + entry_block.body[:]
                    #entry_block.body = entry_block.body[:inst_num] + before_start + [or_start] + entry_block.body[inst_num:]
                    #exit_block.body = [or_end] + priv_restores + exit_block.body
                    exit_block.body = [or_end] + priv_restores + keep_alive + exit_block.body
                    #exit_block.body = [or_end] + exit_block.body

                    break

        assert(found_loop)

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

    def map_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit map_clause", args, type(args), args[0])
        if args[0] in ["to", "from", "alloc", "tofrom"]:
            map_type = args[0].upper()
            var_list = args[1:]
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

    def target_directive(self, args):
        clauses = args[1:]
        tag_clauses = self.flatten(clauses)

        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]

        if config.DEBUG_OPENMP >= 1:
            print("visit target_directive", args, type(args))
        target_num = OpenmpVisitor.target_num
        OpenmpVisitor.target_num += 1
        or_start = openmp_region_start([openmp_tag("DIR.OMP.TARGET", target_num)] + tag_clauses, 0, self.loc)
        or_end   = openmp_region_end(or_start, [openmp_tag("DIR.OMP.END.TARGET", target_num)], self.loc)
        sblk.body = [or_start] + sblk.body[:]
        eblk.body = [or_end]   + eblk.body[:]

    def target_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit target_clause", args, type(args), args[0])
        return args[0]

    def variable_array_section_list(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit variable_array_section_list", args, type(args))
        if len(args) == 1:
            return args
        else:
            args[0].append(args[1])
            return args[0]

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

    def NOWAIT(self, args):
        return "nowait"

    # Don't need a rule for TASK.

    def task_directive(self, args):
        clauses = args[1:]
        tag_clauses = []
        for clause in clauses:
            tag_clauses.extend(clause)

        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        start_tags = [openmp_tag("DIR.OMP.TASK")] + tag_clauses
        end_tags   = [openmp_tag("DIR.OMP.END.TASK")]
        keep_alive = []

        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)
        for itr in inputs_to_region:
            if config.DEBUG_OPENMP >= 1:
                print("input_to_region:", itr)
            start_tags.append(openmp_tag("QUAL.OMP.FIRSTPRIVATE", ir.Var(scope, itr, self.loc)))
        for itr in def_but_live_out:
            if config.DEBUG_OPENMP >= 1:
                print("def_but_live_out:", itr)
            itr_var = ir.Var(scope, itr, self.loc)
            start_tags.append(openmp_tag("QUAL.OMP.SHARED", itr_var))
            keep_alive.append(ir.Assign(itr_var, itr_var, self.loc))
        for ptr in private_to_region:
            if config.DEBUG_OPENMP >= 1:
                print("private_to_region:", ptr)
            start_tags.append(openmp_tag("QUAL.OMP.PRIVATE", ir.Var(scope, ptr, self.loc)))

        if config.DEBUG_OPENMP >= 1:
            print("visit task_directive", args, type(args), tag_clauses)
        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = [or_start] + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        eblk.body = [or_end]   + keep_alive + eblk.body[:]

    def task_clause(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit task_clause", args, type(args), args[0])
        return args[0]

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
        (_, var_list) = args
        ret = []
        for var in var_list:
            ret.append(openmp_tag("QUAL.OMP.SHARED", var))
        return ret

    # Don't need a rule for SHARED.

    def copyin_clause(self, args):
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

    # Don't need a rule for PARALLEL_CONSTRUCT.

    def parallel_directive(self, args):
        sblk = self.blocks[self.blk_start]
        eblk = self.blocks[self.blk_end]
        scope = sblk.scope

        clauses = []
        default_shared = True
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_directive", args, type(args))
        incoming_clauses = [remove_indirections(x) for x in args[1:]]
        # Process all the incoming clauses which can be in singular or list form
        # and flatten them to a list of openmp_tags.
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

        if config.DEBUG_OPENMP >= 1:
            for clause in clauses:
                print("final clause:", clause)

        # Get a dict mapping variables explicitly mentioned in the data clauses above to their openmp_tag.
        vars_in_explicit_clauses, explicit_privates = self.get_explicit_vars(clauses)
        if config.DEBUG_OPENMP >= 1:
            print("vars_in_explicit_clauses:", vars_in_explicit_clauses, type(vars_in_explicit_clauses))

        # Do an analysis to get variable use information coming into and out of the region.
        inputs_to_region, def_but_live_out, private_to_region = self.find_io_vars(self.body_blocks)

        # Returns a dict of private clause variables and their potentially SSA form at the end of the region.
        clause_privates = self.get_clause_privates(clauses, def_but_live_out, scope, self.loc)
        priv_saves = []
        priv_restores = []
        # Numba typing is not aware of OpenMP semantics, so for private variables we save the value
        # before entering the region and then restore it afterwards but we have to restore it to the SSA
        # version of the variable at that point.
        for cp in clause_privates:
            cpvar = ir.Var(scope, cp, self.loc)
            cplovar = ir.Var(scope, clause_privates[cp], self.loc)
            save_var = ir.Var(scope, mk_unique_var("$"+cp), self.loc)
            priv_saves.append(ir.Assign(cpvar, save_var, self.loc))
            priv_restores.append(ir.Assign(save_var, cplovar, self.loc))

        # Remove variables the user explicitly added to a clause from the auto-determined variables.
        # This will also treat SSA forms of vars the same as their explicit Python var clauses.
        self.remove_explicit_from_io_vars(inputs_to_region, def_but_live_out, private_to_region, vars_in_explicit_clauses, clauses, scope, self.loc)

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

        start_tags = [openmp_tag("DIR.OMP.PARALLEL")]
        end_tags = [openmp_tag("DIR.OMP.END.PARALLEL")]
        keep_alive = []
        self.add_variables_to_start(scope, vars_in_explicit_clauses, clauses, True, start_tags, keep_alive, inputs_to_region, def_but_live_out, private_to_region)

        or_start = openmp_region_start(start_tags, 0, self.loc)
        or_end   = openmp_region_end(or_start, end_tags, self.loc)
        sblk.body = priv_saves + [or_start] + sblk.body[:]
        #eblk.body = [or_end]   + eblk.body[:]
        eblk.body = [or_end] + priv_restores + keep_alive + eblk.body[:]

    def parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit parallel_clause", args, type(args))
        return val

    def unique_parallel_clause(self, args):
        (val,) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit unique_parallel_clause", args, type(args))
        assert(isinstance(val, openmp_tag))
        return val

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

    def linear_expr(self, args):
        (_, var, step) = args
        if config.DEBUG_OPENMP >= 1:
            print("visit linear_expr", args, type(args))
        return openmp_tag("QUAL.OMP.LINEAR", [var, step])

    def ORDERED(self, args):
        if config.DEBUG_OPENMP >= 1:
            print("visit ordered", args, type(args))
        return "ordered"

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
        return "RUNTIME"

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
            return args
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
    openmp_construct: parallel_construct
                    | parallel_for_construct
                    | for_construct
                    | single_construct
                    | task_construct
                    | target_construct
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
    teams_distribute_parallel_for_simd_clause: target_clause
                                             | teams_distribute_parallel_for_simd_clause
    for_simd_construct: for_simd_directive
    for_simd_directive: FOR SIMD [for_simd_clause*]
    for_simd_clause: for_clause
                   | simd_clause
    parallel_for_simd_construct: parallel_for_simd_directive
    parallel_for_simd_directive: PARALLEL FOR SIMD [parallel_for_simd_clause*]
    parallel_for_simd_clause: parallel_for_clause
                            | simd_clause
    target_data_construct: target_data_directive
    target_data_directive: TARGET DATA [target_data_clause*]
    DATA: "data"
    target_data_clause: device_clause
                      | map_clause
                      | if_clause
    device_clause: "device" "(" const_num_or_var ")"
    map_clause: "map" "(" [map_type ":"] variable_array_section_list ")"
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
    target_construct: target_directive
    target_directive: TARGET [target_clause*]
    target_clause: device_clause
                 | map_clause
                 | if_clause
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
    single_construct: single_directive
    single_directive: SINGLE [single_clause*]
    SINGLE: "single"
    single_clause: unique_single_clause
                 | data_privatization_clause
                 | data_privatization_in_clause
                 | NOWAIT
    unique_single_clause: copyprivate_clause
    NOWAIT: "nowait"
    simd_construct: simd_directive
    simd_directive: SIMD [simd_clause*]
    SIMD: "simd"
    simd_clause: collapse_expr
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
    collapse_expr: COLLAPSE "(" const_num_or_var ")"
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
                      | DEPEND "(" dependence_type ":" variable_array_section_list ")"
    DEPEND: "depend"
    FINAL: "final"
    UNTIED: "untied"
    MERGEABLE: "MERGEABLE"
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
                     | collapse_expr
    LINEAR: "linear"
    linear_clause: LINEAR "(" var_list ":" const_num_or_var ")"
                 | LINEAR "(" var_list ")"
    ORDERED: "ordered"
    sched_no_expr: SCHEDULE "(" schedule_kind ")"
    sched_expr: SCHEDULE "(" schedule_kind "," const_num_or_var ")"
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
    TASKGROUP: "taskgroup"
    cancel_directive: "cancel" construct_type_clause [if_clause]
    ordered_directive: ORDERED
    ordered_construct: ordered_directive
    flush_vars: "(" var_list ")"
    flush_directive: "flush" [flush_vars]
    region_phrase: "(" PYTHON_NAME ")"
    master_construct: master_directive
    master_directive: "master"
    dependence_type: IN
                   | OUT
                   | INOUT
    IN: "in"
    OUT: "out"
    INOUT: "inout"

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
                    | target_teams_distribute_parallel_for_simd_construct
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

    sblk = blocks[blk_start]
    scope = sblk.scope
    loc = sblk.loc
    sblk.body = sblk.body[1:]

    args = extra["args"]
    arg = args[0]
    parse_res = openmp_parser.parse(arg.value)
    if config.DEBUG_OPENMP >= 1:
        print("args:", args, type(args))
        print("arg:", arg, type(arg), arg.value, type(arg.value))
        print(parse_res.pretty())
    visitor = OpenmpVisitor(func_ir, blocks, blk_start, blk_end, body_blocks, loc, state)
    try:
        visitor.transform(parse_res)
    except VisitError as e:
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

ffi = FFI()
ffi.cdef('void omp_set_num_threads(int num_threads);')
ffi.cdef('int omp_get_thread_num(void);')
ffi.cdef('int omp_get_num_threads(void);')
ffi.cdef('double omp_get_wtime(void);')
ffi.cdef('void omp_set_dynamic(int num_threads);')
ffi.cdef('void omp_set_nested(int nested);')
ffi.cdef('void omp_set_max_active_levels(int levels);')
ffi.cdef('int omp_get_max_active_levels(void);')
ffi.cdef('int omp_get_max_threads(void);')
ffi.cdef('int omp_get_num_procs(void);')
ffi.cdef('int omp_in_parallel(void);')
ffi.cdef('int omp_get_thread_limit(void);')
ffi.cdef('int omp_get_supported_active_levels(void);')
ffi.cdef('int omp_get_level(void);')
ffi.cdef('int omp_get_active_level(void);')
ffi.cdef('int omp_get_ancestor_thread_num(int level);')
ffi.cdef('int omp_get_team_size(int level);')
ffi.cdef('int omp_in_final(void);')
ffi.cdef('int omp_get_proc_bind(void);')
ffi.cdef('int omp_get_num_places(void);')
ffi.cdef('int omp_get_place_num_procs(int place_num);')
ffi.cdef('int omp_get_place_num(void);')
ffi.cdef('int omp_set_default_device(int device_num);')
ffi.cdef('int omp_get_default_device(void);')
ffi.cdef('int omp_get_num_devices(void);')
ffi.cdef('int omp_get_device_num(void);')
ffi.cdef('int omp_get_num_teams(void);')
ffi.cdef('int omp_is_initial_device(void);')
ffi.cdef('int omp_get_initial_device(void);')

C = ffi.dlopen(None)
omp_set_num_threads = C.omp_set_num_threads
omp_get_thread_num = C.omp_get_thread_num
omp_get_num_threads = C.omp_get_num_threads
omp_get_wtime = C.omp_get_wtime
omp_set_dynamic = C.omp_set_dynamic
omp_set_nested = C.omp_set_nested
omp_set_max_active_levels = C.omp_set_max_active_levels
omp_get_max_active_levels = C.omp_get_max_active_levels
omp_get_max_threads = C.omp_get_max_threads
omp_get_num_procs = C.omp_get_num_procs
omp_in_parallel = C.omp_in_parallel
omp_get_thread_limit = C.omp_get_thread_limit
omp_get_supported_active_levels = C.omp_get_supported_active_levels
omp_get_level = C.omp_get_level
omp_get_active_level = C.omp_get_active_level
omp_get_ancestor_thread_num = C.omp_get_ancestor_thread_num
omp_get_team_size = C.omp_get_team_size
omp_in_final = C.omp_in_final
omp_get_proc_bind = C.omp_get_proc_bind
omp_get_num_places = C.omp_get_num_places
omp_get_place_num_procs = C.omp_get_place_num_procs
omp_get_place_num = C.omp_get_place_num
omp_set_default_device = C.omp_set_default_device
omp_get_default_device = C.omp_get_default_device
omp_get_num_devices = C.omp_get_num_devices
omp_get_device_num = C.omp_get_device_num
omp_get_num_teams = C.omp_get_num_teams
omp_is_initial_device = C.omp_is_initial_device
omp_get_initial_device = C.omp_get_initial_device
